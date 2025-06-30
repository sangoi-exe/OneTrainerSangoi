import copy
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseStableDiffusionXLSetup import CapturingAttnProcessor
from modules.sangoi.LogFun import logFun
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.memory_util import TorchMemoryRecorder
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

from PIL.Image import Image
from tqdm import tqdm

from modules.sangoi.TokenGradientAnalyzer import TokenGradientAnalyzer


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard_subprocess: subprocess.Popen
    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]
    
    # params do pause
    is_paused: bool
    pause_request_locked: bool  # Para travar o switch da UI
    pause_requested_at_epoch_end: bool

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, get_string_timestamp()))
        if config.tensorboard:
            tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")

            tensorboard_args = [
                tensorboard_executable,
                "--logdir",
                tensorboard_log_dir,
                "--port",
                str(config.tensorboard_port),
                "--samples_per_plugin=images=100,scalars=10000",
            ]

            if self.config.tensorboard_expose:
                tensorboard_args.append("--bind_all")

            self.tensorboard_subprocess = subprocess.Popen(tensorboard_args)

        self.model = None
        self.one_step_trained = False

        self.grad_hook_handles = []

        self.is_paused = False
        self.pause_request_locked = False
        self.pause_requested_at_epoch_end = False

        self._steps_per_epoch = None
        self.token_analyzer = None

    def _handle_pause_logic(self):
        """Executa a lógica de pausa, movendo o modelo e esperando."""
        if not self.is_paused:  # Segurança extra
            return

        logFun("Iniciando Pausa...", lvl="LOOP")
        self.callbacks.on_update_status("Pausing... Moving model to CPU")
        try:
            self.model.to(self.temp_device)
            self.model.eval()
            torch_gc()
            
            logFun(f"Modelo movido para {self.temp_device}. VRAM liberada.", lvl="success")
            
            self.callbacks.on_update_status(f"Paused. Model on {self.temp_device}. Toggle switch to resume.")

            if hasattr(self.callbacks, 'on_pause_initiated'):
                self.callbacks.on_pause_initiated()

            while self.is_paused:
                if self.commands.get_stop_command():
                    logFun("Comando STOP recebido durante a pausa. Interrompendo.", lvl="warning")
                    self.is_paused = False
                    break

                if self.commands.get_and_reset_resume_request():
                    logFun("Comando RESUME recebido.", lvl="info")
                    self.is_paused = False 
                    self.pause_request_locked = False 

                    if hasattr(self.callbacks, 'on_resume_started'):
                        self.callbacks.on_resume_started()
                    break

                time.sleep(1.0)

            if not self.commands.get_stop_command():  # Só retoma se não for parar
                logFun("Retomando treinamento...", lvl="info")
                self.callbacks.on_update_status("Resuming... Moving model to GPU")
                try:
                    self.model_setup.setup_train_device(self.model, self.config)
                    torch_gc()
                    logFun(f"Modelo movido de volta para {self.config.train_device}.", lvl="success")
                    self.callbacks.on_update_status("Training resumed.")

                    if hasattr(self.callbacks, 'on_resume_completed'):
                        self.callbacks.on_resume_completed()

                except Exception as e:
                    logFun(f"Erro ao mover modelo de volta para GPU: {e}", lvl="error")
                    traceback.print_exc()
                    # Tentar continuar mesmo assim? Ou parar? Por segurança, parar.
                    self.commands.stop()
            else:
                logFun("Retomada cancelada devido ao comando STOP.", lvl="warning")

        except Exception as e:
            logFun(f"Erro durante o processo de pausa/retomada: {e}", lvl="error")
            traceback.print_exc()
            self.is_paused = False
            self.pause_request_locked = False

            self.commands.stop()
            self.callbacks.on_update_status(f"Error during pause/resume: {e}")

    @staticmethod
    def stop_grad_outside_mask(tensor: torch.Tensor, mask_bf16: torch.Tensor) -> None:
        """
        Mantém o forward intacto (contexto total) e zera gradiente fora da máscara.
        * `tensor`: saída bruta do modelo (bf16/fp16/fp32).
        * `mask`  : mesma shape espacial, dtype float/bool (1 = região de interesse).
        """
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * mask_bf16       # mesmo dtype → sem crash
        tensor.register_hook(_hook)

    @staticmethod
    def prepare_mask(
        mask: torch.Tensor,
        ref: torch.Tensor,
        thresh: float = 0.5
        ) -> torch.Tensor:
        """Binariza + broadcasta máscara para ter shape/dtype de `ref`."""
        m = (mask > thresh).to(dtype=ref.dtype, device=ref.device)
        if m.ndim < ref.ndim:            # [B,H,W] → [B,1,H,W]
            m = m.unsqueeze(1)
        if m.shape[1] == 1 and ref.shape[1] != 1:
            m = m.expand(ref.shape[0], ref.shape[1], *m.shape[2:])
        return m

    def start(self):
        self.__save_config_to_workspace()

        if self.config.clear_cache_before_training and self.config.latent_caching:
            self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        self.callbacks.on_update_status("loading the model")
        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
        )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)

        print("Ativando Token Gradient Analyzer.")
        self.token_analyzer = TokenGradientAnalyzer(
            tokenizer_l=self.model.tokenizer_1,
            tokenizer_g=self.model.tokenizer_2,
            out_dir=os.path.join(self.config.workspace_dir, "token_affinity_reports"),
            config=self.config
        )
        self.token_analyzer.start_analysis_hooks(self.model)

        self.model.eval()
        torch_gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model.train_progress
        )
        
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()
        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model.train_progress, is_validation=True
            )

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(), f, indent=4)

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.config.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith('epoch-') or filename in ['image', 'text']):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [dirpath for dirpath in os.listdir(backup_dirpath) if
                os.path.isdir(os.path.join(backup_dirpath, dirpath))],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    print(f"Could not delete old rolling backup {dirpath}")

        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def __sample_loop(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_config_list: list[SampleConfig],
            folder_postfix: str = "",
            image_format: ImageFormat = ImageFormat.JPG,
            is_custom_sample: bool = False,
    ):
        for i, sample_config in enumerate(sample_config_list):
            if sample_config.enabled:
                try:
                    safe_prompt = path_util.safe_filename(sample_config.prompt)

                    if is_custom_sample:
                        sample_dir = os.path.join(
                            self.config.workspace_dir,
                            "samples",
                            "custom",
                        )
                    else:
                        sample_dir = os.path.join(
                            self.config.workspace_dir,
                            "samples",
                            f"{str(i)} - {safe_prompt}{folder_postfix}",
                        )

                    sample_path = os.path.join(
                        sample_dir,
                        f"{get_string_timestamp()}-training-sample-{train_progress.filename_string()}{image_format.extension()}"
                    )

                    def on_sample_default(image: Image):
                        if self.config.samples_to_tensorboard:
                            self.tensorboard.add_image(
                                f"sample{str(i)} - {safe_prompt}", pil_to_tensor(image),  # noqa: B023
                                train_progress.global_step
                            )
                        self.callbacks.on_sample_default(image)

                    def on_sample_custom(image: Image):
                        self.callbacks.on_sample_custom(image)

                    on_sample = on_sample_custom if is_custom_sample else on_sample_default
                    on_update_progress = self.callbacks.on_update_sample_custom_progress if is_custom_sample else self.callbacks.on_update_sample_default_progress

                    self.model.to(self.temp_device)
                    self.model.eval()

                    sample_config = copy.copy(sample_config)
                    sample_config.from_train_config(self.config)

                    self.model_sampler.sample(
                        sample_config=sample_config,
                        destination=sample_path,
                        image_format=self.config.sample_image_format,
                        on_sample=on_sample,
                        on_update_progress=on_update_progress,
                    )
                except Exception:
                    traceback.print_exc()
                    print("Error during sampling, proceeding without sampling")

                torch_gc()

    def __sample_during_training(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("sampling")

        is_custom_sample = False
        if not sample_params_list:
            if self.config.samples is not None:
                sample_params_list = self.config.samples
            else:
                with open(self.config.sample_definition_file_name, 'r') as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values().from_dict(samples[i])
                    sample_params_list = samples
        else:
            is_custom_sample = True

        if self.model.ema:
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            image_format=self.config.sample_image_format,
            is_custom_sample=is_custom_sample,
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if an ema model exists
        if self.model.ema and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                image_format=self.config.sample_image_format,
                folder_postfix=" - no-ema",
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation)

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model, validation_batch, self.config, train_progress)
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config)

                # since validation batch size = 1
                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                # check and fix collision to display both graphs in tensorboard
                if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                                            average_loss,
                                            train_progress.global_step)

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar("loss/validation_step/total_average",
                                            total_average_loss,
                                            train_progress.global_step)

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def backup(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def save(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}"
        )
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        if self.config.skip_sample_on_train_start and train_progress.epoch == 0 and train_progress.epoch_step == 0:
            return False
        return self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(self, scaler):
        if self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating fused_back_pass with gradient_accumulation_steps > 1 does not reduce VRAM usage.")

            for param_group in self.model.optimizer.param_groups:
                for i, parameter in enumerate(param_group["params"]):
                    # TODO: Find a better check instead of "parameter.requires_grad".
                    #       This will break if the some parameters don't require grad during the first training step.
                    if parameter.requires_grad:
                        if scaler:
                            def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                                if self.__is_update_step(self.model.train_progress):
                                    scaler.unscale_parameter_(tensor, self.model.optimizer)
                                    if self.config.clip_grad_norm is not None:
                                        nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                                    scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                                    tensor.grad = None
                        else:
                            def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                                if self.__is_update_step(self.model.train_progress):
                                    if self.config.clip_grad_norm is not None:
                                        nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                                    self.model.optimizer.step_parameter(tensor, param_group, i)
                                    tensor.grad = None

                        handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                        self.grad_hook_handles.append(handle)

    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
    @staticmethod
    def dissect_batch(batch: dict, step: int):
        """
        Uma função de depuração brutalmente honesta para inspecionar
        o conteúdo de um dicionário de batch do PyTorch.
        """
        print("\n" + "="*40)
        print(f"AUTOPSY OF BATCH - STEP: {step}")
        print("="*40)

        if not isinstance(batch, dict):
            print(f"  ERROR: Batch is not a dictionary, but a {type(batch)}.")
            print("="*40 + "\n")
            return

        for key, value in batch.items():
            print(f"\n--- Key: '{key}' ---")
            
            if isinstance(value, torch.Tensor):
                print(f"  Type: torch.Tensor")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Device: {value.device}")
                print(f"  Requires Grad: {value.requires_grad}")
                # Mostra um pequeno trecho do tensor para dar uma ideia do conteúdo
                # Flatten para lidar com qualquer número de dimensões
                preview = value.flatten()[:8].to(torch.float32).cpu().numpy()
                print(f"  Preview: {preview}")

            elif isinstance(value, list):
                print(f"  Type: list")
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    first_item = value[0]
                    print(f"  Type of first item: {type(first_item)}")
                    if isinstance(first_item, torch.Tensor):
                        print(f"  Shape of first item: {first_item.shape}")
                    elif isinstance(first_item, str):
                        # Limita o tamanho da string para não poluir o log
                        print(f"  Preview of first item: '{first_item[:100]}...'")
                    else:
                        print(f"  Preview of first item: {first_item}")
            
            elif isinstance(value, dict):
                print(f"  Type: dict")
                print(f"  Keys: {list(value.keys())}")

            else:
                print(f"  Type: {type(value)}")
                print(f"  Value: {value}")

        print("\n" + "="*40)
        print("END OF AUTOPSY")
        print("="*40 + "\n")

    def train(self):
        train_device = torch.device(self.config.train_device)
        train_progress = self.model.train_progress
        
        if self.config.only_cache:
            self.callbacks.on_update_status("caching")
            for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
                self.data_loader.get_data_set().start_next_epoch()
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = 0.0
        ema_loss = None
        for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):

            if self.is_paused:
                logFun(f"Treino iniciado em estado PAUSADO (Epoch {train_progress.epoch}). Aguardando resume...", lvl="warning")
                self._handle_pause_logic()
                if self.commands.get_stop_command():  # Se o stop foi dado durante a pausa inicial
                    logFun("Comando STOP ativo após pausa inicial. Encerrando.", lvl="warning")
                    break

            self.callbacks.on_update_status("starting epoch/caching")

            if self.config.latent_caching:
                self.data_loader.get_data_set().start_next_epoch()
                self.model_setup.setup_train_device(self.model, self.config)
            else:
                self.model_setup.setup_train_device(self.model, self.config)
                self.data_loader.get_data_set().start_next_epoch()

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()
            step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step", total=current_epoch_length, initial=train_progress.epoch_step)


            # else:
            #     print("[GenericTrainer] Ativando requires_grad para embeddings (análise de token)..")
            #     emb_layer_l = self.model.text_encoder_1.get_input_embeddings()
            #     if not emb_layer_l.weight.requires_grad:
            #       emb_layer_l.weight.requires_grad_(True)
                
            #     emb_layer_g = self.model.text_encoder_2.get_input_embeddings()
            #     if not emb_layer_g.weight.requires_grad:
            #       emb_layer_g.weight.requires_grad_(True)
            
            train_progress.set_steps_per_epoch(self.data_loader.get_data_set().approximate_length())
            self._steps_per_epoch = train_progress.steps_per_epoch
            train_progress.set_total_steps(self._steps_per_epoch * self.config.epochs)

            for batch in step_tqdm:
                if self.__needs_sample(train_progress) or self.commands.get_and_reset_sample_default_command():
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )

                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:
                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    transferred_to_temp_device = False

                    if self.commands.get_and_reset_backup_command():
                        self.model.to(self.temp_device)
                        self.backup(train_progress, True, step_tqdm.write)
                        transferred_to_temp_device = True

                    if self.commands.get_and_reset_save_command():
                        self.model.to(self.temp_device)
                        self.save(train_progress, True, step_tqdm.write)
                        transferred_to_temp_device = True

                    if transferred_to_temp_device:
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("training")

                # 1. ATIVAÇÃO DE GRADIENTE PARA O MODO CACHE (Custo insignificante)
                if self.token_analyzer and not self.config.train_text_encoder_or_embedding():
                    
                    hidden_states_l = batch.get('text_encoder_1_hidden_state')
                    if hidden_states_l is not None:
                        hidden_states_l.requires_grad_(True)
                    
                    hidden_states_g = batch.get('text_encoder_2_hidden_state')
                    if hidden_states_g is not None:
                        hidden_states_g.requires_grad_(True)
                    
                    pooled_output_g = batch.get('text_encoder_2_pooled_state')
                    if pooled_output_g is not None:
                        pooled_output_g.requires_grad_(True)

                with TorchMemoryRecorder(enabled=False):
                    # Prepara o analisador para a coleta de dados
                    if self.token_analyzer:
                        self.token_analyzer.set_pending_analysis(train_progress.global_step, batch)
                        # Instala os hooks de atenção se for um passo de análise de atenção
                        if self.token_analyzer.enable_attn_report:
                            self.token_analyzer.start_analysis_hooks(self.model)

                    # 2. FORWARD PASS
                    model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                    loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config, train_progress, self.tensorboard)
                    
                    # 3. BACKWARD PASS
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                    # 4. ANÁLISE DE GRADIENTE (O PONTO CRÍTICO)
                    # DEPOIS do backward, ANTES do step/zero_grad.
                    if self.token_analyzer:
                        self.token_analyzer.analyze_gradients_after_backward(self.model)

                    # 5. ATUALIZAÇÃO DOS PESOS
                    self.model.optimizer.step()
                    self.model.optimizer.zero_grad(set_to_none=True)

                    # 6. ANÁLISE DE ATENÇÃO E LIMPEZA (Final do passo)
                    if self.token_analyzer:
                        self.token_analyzer.analyze_attention_after_step(self.model, train_progress.epoch)
                        # Desinstala os hooks de atenção
                        if self.token_analyzer.enable_attn_report:
                            self.token_analyzer.stop_analysis_hooks(self.model)

                    has_gradient = True
                    accumulated_loss += loss.item()

                    if self.__is_update_step(train_progress):
                        if scaler and self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()

                        lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                        self.model.optimizer.zero_grad(set_to_none=True)
                        has_gradient = False

                        self.model_setup.report_to_tensorboard(
                            self.model, self.config, lr_scheduler, self.tensorboard
                        )

                        self.tensorboard.add_scalar("loss/train_step", accumulated_loss, train_progress.global_step)
                        ema_loss = ema_loss or accumulated_loss
                        ema_loss = (ema_loss * 0.99) + (accumulated_loss * 0.01)
                        step_tqdm.set_postfix({
                            'loss': accumulated_loss,
                            'smooth loss': ema_loss,
                        })
                        self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)
                        accumulated_loss = 0.0

                        # vai tomar no cu, Nerogar
                        # self.model_setup.after_optimizer_step(self.model, self.config, train_progress)
                        if self.model.ema:
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay",
                                self.model.ema.get_current_decay(update_step),
                                train_progress.global_step
                            )
                            self.model.ema.step(
                                self.parameters,
                                update_step
                            )

                        self.one_step_trained = True

                if self.config.validation:
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    return
            
            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                return
            
            if self.commands.get_and_reset_pause_request():
                logFun(f"Requisição de PAUSA recebida. Será executada ao final da Epoch {train_progress.epoch -1}.", lvl="info")
                self.pause_requested_at_epoch_end = True
                self.pause_request_locked = True
                if hasattr(self.callbacks, 'on_pause_request_accepted'):
                    self.callbacks.on_pause_request_accepted()

            # 2. Executar a pausa se foi agendada
            if self.pause_requested_at_epoch_end and not self.is_paused:
                self.is_paused = True  # Marca como pausado
                self.pause_requested_at_epoch_end = False  # Limpa a flag de agendamento
                # A trava (pause_request_locked) continua TRUE até o resume
                # Chama a função que move o modelo e entra no loop de espera
                self._handle_pause_logic()


    def end(self):
        if self.one_step_trained:
            self.model.to(self.temp_device)

            if self.config.backup_before_save:
                self.backup(self.model.train_progress)
            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            self.callbacks.on_update_status("saving the final model")

            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=False)

            # toma bem no meio do cu do nerogar, tem função de save até dentro do rabo dele
            # aí tive que fazer uma gambiarra aqui pra evitar que um safetensor seja sobrescrito
            # esse doente consegue ser muito inteligente por criar o OT, mas um ANIMAL por não prever esse tipo de coisa
            output_model_destination = os.path.join(
                self.config.workspace_dir,
                "output",
                f"{self.config.output_model_destination}_{get_string_timestamp(fmt='%Hh%M_%d-%b-%Y')}{self.config.output_model_format.file_extension()}"
            )
            
            print("Saving " + output_model_destination)

            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=output_model_destination,
                dtype=self.config.output_dtype.torch_dtype()
            )

        elif self.model is not None:
            self.model.to(self.temp_device)

        if self.token_analyzer:
          self.token_analyzer.stop_analysis_hooks(self.model)

        self.tensorboard.close()

        if self.config.tensorboard:
            self.tensorboard_subprocess.kill()

        for handle in self.grad_hook_handles:
            handle.remove()
