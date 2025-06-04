import collections
import contextlib
import copy
import json
import os
import shutil
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Optional

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.sangoi.DataRecorder import DataRecorder
from modules.sangoi.logFun import (
    ProgressContext,
    cleanup_global_progress,
    init_global_progress,
    logFun,
    set_logfun_console,
)
from modules.sangoi.TrainGPS import TrainGPS
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.memory_util import TorchMemoryRecorder
from modules.util.TensorBoardManager import TensorBoardManager
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torchvision.transforms.functional import pil_to_tensor

import huggingface_hub
import tqdm
from pytorch_msssim import ssim
from requests.exceptions import ConnectionError
from rich.align import Align
from rich.console import Console as RichConsole
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, track
from rich.table import Table
from rich.text import Text

tqdm.tqdm = lambda iterable, **kwargs: track(iterable, **kwargs)


def format_time_delta(seconds: float) -> str:
    if seconds < 0 or not isinstance(seconds, (int, float)):
        return "??:??"  # Lida com valores inválidos
    seconds = abs(seconds)  # Garante que é positivo para cálculo
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


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

    tensorboard: TensorBoardManager
    recorder: DataRecorder | None
    _temp_recorder_data: collections.defaultdict  # Temporary storage for recorder data before log_step

    # atributos para pause
    is_paused: bool
    pause_request_locked: bool  # Para travar o switch da UI
    pause_requested_at_epoch_end: bool

    _training_live: Live | None = None  # A única instância Live para todo o display
    _global_progress_instance: Optional["Progress"] = None  # Referência ao objeto Progress global do logFun

    _current_step_duration_s: float = 0.0
    _epoch_time_elapsed_s: float = 0.0
    _avg_step_time_epoch_s: float = 0.0
    _ema_step_time_s: float | None = None  # Para média móvel exponencial
    _ema_alpha: float = 0.05  # Ajuste para mais ou menos suavização (menor = mais suave)
    _total_training_time_start_s: float | None = None  # Para tempo total de treino

    _epoch_start_time_s: float | None = None  # Para calcular tempo da epoch e ETA
    _num_total_epochs: int = 0
    _steps_per_epoch: int = 0

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        self.tensorboard = TensorBoardManager(
            log_dir=os.path.join(
                tensorboard_log_dir,
                f"{config.save_filename_prefix}{get_string_timestamp()}",
            )
        )
        if config.tensorboard:
            super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []

        self.is_paused = False
        self.recorder = None
        self.train_dtype = None
        self.pause_request_locked = False
        self.pause_requested_at_epoch_end = False
        self.train_device = torch.device(self.config.train_device)

        self.console = RichConsole()
        set_logfun_console(self.console)

        self._num_total_epochs = config.epochs
        self.timestep_perf = torch.zeros(1000, device=self.config.train_device)
        self.timestep_count = torch.zeros_like(self.timestep_perf, dtype=torch.long)
        self.ema_alpha = getattr(self.config, "timestep_perf_ema_alpha", 0.9)
        self.burn_in_steps = getattr(self.config, "timestep_burn_in_steps", 1000)

    def _create_training_display_content(
        self, current_loss: float | None = None, current_ema_loss: float | None = None
    ) -> Panel:
        """Cria o conteúdo do painel principal de treinamento com dados atualizados."""

        tp = self.model.train_progress

        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")

        epoch_str = f"Epoch {tp.epoch + 1}/{self._num_total_epochs}" if tp else f"Epoch ?/{self._num_total_epochs}"
        step_str = f"Step {tp.epoch_step + 1}/{self._steps_per_epoch}" if tp and self._steps_per_epoch > 0 else "Step ?"
        global_step_str = f"Global {tp.global_step + 1}" if tp else "Global ?"
        table.add_row(
            Text(epoch_str, style="bold cyan"),
            Text(step_str, style="bold cyan"),
            Text(global_step_str, style="dim cyan"),
        )

        loss_str = f"Loss {current_loss:.8f}" if current_loss is not None else "Loss N/A"
        ema_loss_str = f"Smooth {current_ema_loss:.8f}" if current_ema_loss is not None else "Smooth N/A"
        table.add_row(Text(loss_str, style="yellow"), Text(ema_loss_str, style="bright_yellow"), Text(""))

        step_time_disp = f"StepTime {self._current_step_duration_s:.2f}s"

        effective_avg_step_time = self._avg_step_time_epoch_s
        if self._ema_step_time_s is not None and self._ema_step_time_s > 0:
            effective_avg_step_time = self._ema_step_time_s

        avg_step_disp = (
            f"AvgStep {effective_avg_step_time:.2f}s/it" if effective_avg_step_time > 0 else "AvgStep Calc..."
        )
        epoch_elapsed_disp = f"EpochElap {format_time_delta(self._epoch_time_elapsed_s)}"

        eta_epoch_disp = "ETAEpoch ???"
        if tp and self._steps_per_epoch > 0 and effective_avg_step_time > 0:
            remaining_steps = max(self._steps_per_epoch - (tp.epoch_step + 1), 0)
            eta_s = remaining_steps * effective_avg_step_time
            eta_epoch_disp = f"ETAEpoch {format_time_delta(eta_s)}"

        eta_total_disp = "ETATotal ???"
        if tp and self._steps_per_epoch > 0 and effective_avg_step_time > 0:
            total_steps = self._num_total_epochs * self._steps_per_epoch
            completed_steps = tp.global_step + 1
            remaining_steps_total = max(total_steps - completed_steps, 0)
            eta_s_total = remaining_steps_total * effective_avg_step_time
            eta_total_disp = f"ETATotal {format_time_delta(eta_s_total)}"

        total_time_str = ""
        if self._total_training_time_start_s is not None:
            total_elapsed = time.monotonic() - self._total_training_time_start_s
            total_time_str = f"TotalRun {format_time_delta(total_elapsed)}"

        table.add_row(
            Text(step_time_disp, style="green"),
            Text(avg_step_disp, style="blue"),
            Text(epoch_elapsed_disp, style="magenta"),
        )
        table.add_row(
            Text(eta_epoch_disp, style="magenta"),
            Text(eta_total_disp, style="magenta"),
            Text(total_time_str, style="dim white"),
        )

        if self.recorder:
            table.add_row(Text("Recorder", style="yellow"), Text("Ativo", style="yellow"), Text(""))

        panel = Panel(
            Align.center(table), title="[bold cyan]OneTrainer - Status do Treinamento[/bold cyan]", border_style="cyan"
        )
        return panel

    def _get_combined_display_renderable(
        self, current_loss: float | None = None, current_ema_loss: float | None = None
    ) -> Group:
        """
        Combina o painel de status principal de treinamento e o objeto Progress global
        em um único renderable Group.
        """
        # Garante que o objeto _global_progress_instance esteja inicializado
        if self._global_progress_instance is None:
            self._global_progress_instance = init_global_progress()  # Isso agora *apenas cria* o objeto Progress

        # Conteúdo do painel de status
        status_panel_content = self._create_training_display_content(current_loss, current_ema_loss)

        # Cria um Group contendo o painel de status e a barra de progresso global
        # A ordem aqui define a exibição vertical: status_panel em cima, barra de progresso abaixo
        combined_display = Group(
            status_panel_content,
            self._global_progress_instance,  # Este é o objeto Progress do logFun
        )
        return combined_display

    def _setup_training_display(self):
        """Configura o sistema de display de treinamento com Live, exibindo painel e progress bars."""
        if self._training_live is None:
            # Obtém o conteúdo combinado inicial para o Live display
            initial_content = self._get_combined_display_renderable()

            self._training_live = Live(
                initial_content,
                console=self.console,
                refresh_per_second=2,
                screen=False,  # Não toma a tela toda
                transient=False,  # Não limpa ao parar
            )
            # Inicia o display Live uma vez
            self._training_live.start()
            logFun("Sistema de display Rich Live iniciado", lvl="INFO")

    def _update_training_display(self, current_loss: float | None = None, current_ema_loss: float | None = None):
        """Atualiza o display de treinamento."""
        if self._training_live and self._training_live.is_started:
            # Atualiza a instância Live com o novo conteúdo combinado
            updated_content = self._get_combined_display_renderable(current_loss, current_ema_loss)
            self._training_live.update(updated_content)

    def _stop_training_display(self):
        """Para o display de treinamento."""
        if self._training_live and self._training_live.is_started:
            self._training_live.stop()
            self._training_live = None
            self._global_progress_instance = None  # Reseta a referência ao objeto Progress
            logFun("Sistema de display Rich Live finalizado", lvl="INFO")

    def start(self):
        set_logfun_console(self.console)

        self.__save_config_to_workspace()

        if self.config.clear_cache_before_training and self.config.latent_caching:
            self.__clear_cache()

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

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("logging into Hugging Face")
            with contextlib.suppress(ConnectionError):
                huggingface_hub.login(
                    token=self.config.secrets.huggingface_token,
                    new_session=False,
                )

        self.callbacks.on_update_status("loading the model")
        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
        )
        self.model.train_config = self.config
        self.train_dtype = self.model.train_dtype.torch_dtype()
        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config, self.tensorboard)

        # self.model.to(self.temp_device) será que dá pra desativar essa bosta? .. é, dá, filhos da puta, movendo modelo pro temp device a troco de nada

        self.model.eval()
        torch_gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(self.model, self.model.train_progress)
        self._steps_per_epoch = (
            self.data_loader.get_data_set().approximate_length()
        )  # Pega o total de steps para a epoch atual
        print("self._steps_per_epoch", self._steps_per_epoch)

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
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        logFun(f"Limpando diretório de cache {self.config.cache_dir}!", lvl="INFO")
        if os.path.isdir(self.config.cache_dir):
            # ProgressContext já usa o sistema global de progresso
            with ProgressContext("Limpando cache", len(os.listdir(self.config.cache_dir))) as progress:  # Estimar total
                files_to_delete = [
                    f
                    for f in os.listdir(self.config.cache_dir)
                    if os.path.isdir(os.path.join(self.config.cache_dir, f))
                    and (f.startswith("epoch-") or f in ["image", "text"])
                ]

                # A iteração para delete deve ser sobre os arquivos_to_delete
                # O total para ProgressContext também deve ser files_to_delete
                if files_to_delete:
                    # Recria o ProgressContext com o total correto para a lista filtrada
                    # Não é ideal ter duas instâncias, mas a primeira é um placeholder.
                    # Poderíamos filtrar antes de criar o ProgressContext.
                    progress.total = len(files_to_delete)  # Atualiza o total
                    for filename in files_to_delete:
                        path = os.path.join(self.config.cache_dir, filename)
                        shutil.rmtree(path)
                        progress.update(1)
                else:
                    logFun("Nenhum diretório de cache para limpar.", lvl="INFO")

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [
                    dirpath
                    for dirpath in os.listdir(backup_dirpath)
                    if os.path.isdir(os.path.join(backup_dirpath, dirpath))
                ],
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
                        f"{get_string_timestamp()}-training-sample-{train_progress.filename_string()}",
                    )

                    def on_sample_default(sampler_output: ModelSamplerOutput):
                        if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                            self.tensorboard.add_image(
                                f"sample{str(i)} - {safe_prompt}",
                                pil_to_tensor(sampler_output.data),  # noqa: B023
                                train_progress.global_step,
                            )
                        self.callbacks.on_sample_default(sampler_output)

                    def on_sample_custom(sampler_output: ModelSamplerOutput):
                        self.callbacks.on_sample_custom(sampler_output)

                    on_sample = on_sample_custom if is_custom_sample else on_sample_default
                    on_update_progress = (
                        self.callbacks.on_update_sample_custom_progress
                        if is_custom_sample
                        else self.callbacks.on_update_sample_default_progress
                    )

                    self.model.to(self.temp_device)
                    self.model.eval()

                    sample_config = copy.copy(sample_config)
                    sample_config.from_train_config(self.config)

                    self.model_sampler.sample(
                        sample_config=sample_config,
                        destination=sample_path,
                        image_format=self.config.sample_image_format,
                        video_format=self.config.sample_video_format,
                        audio_format=self.config.sample_audio_format,
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
                with open(self.config.sample_definition_file_name, "r") as f:
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

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            # O uso de ProgressContext substitui _rich_progress_footer
            with ProgressContext("Validação", current_epoch_length_validation) as progress:
                for validation_batch in self.validation_data_loader.get_data_loader():
                    if self.__needs_gc(train_progress):
                        torch_gc()

                    with torch.no_grad():
                        model_output_data = self.model_setup.predict(
                            self.model,
                            validation_batch,
                            self.config,
                            train_progress,
                            deterministic=True,
                        )
                        loss_validation = self.model_setup.calculate_loss(
                            self.model, validation_batch, model_output_data, self.config
                        )

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

                    accumulated_loss_per_concept[concept_seed] = (
                        accumulated_loss_per_concept.get(concept_seed, 0) + loss
                    )
                    concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1
                    progress.update(1)

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(
                    f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                    average_loss,
                    train_progress.global_step,
                )

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar(
                    "loss/validation_step/total_average",
                    total_average_loss,
                    train_progress.global_step,
                )

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def backup(
        self,
        train_progress: TrainProgress,
        print_msg: bool = True,
        print_cb: Callable[[str], None] = print,
    ):
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
                logFun("Creating Backup " + backup_path, lvl="LOOP")  # Uso de logFun

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

    def save(
        self,
        train_progress: TrainProgress,
        print_msg: bool = True,
        print_cb: Callable[[str], None] = print,
    ):
        torch_gc()

        self.callbacks.on_update_status("saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}",
        )
        if print_msg:
            logFun("Saving " + save_path, lvl="LOOP")  # Uso de logFun

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
                dtype=self.config.output_dtype.torch_dtype(),
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
        return self.single_action_elapsed(
            "sample_skip_first",
            self.config.sample_skip_first,
            self.config.sample_after_unit,
            train_progress,
        ) and self.repeating_action_needed(
            "sample",
            self.config.sample_after,
            self.config.sample_after_unit,
            train_progress,
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup",
            self.config.backup_after,
            self.config.backup_after_unit,
            train_progress,
            start_at_zero=False,
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first",
            self.config.save_skip_first,
            self.config.save_every_unit,
            train_progress,
        ) and self.repeating_action_needed(
            "save",
            self.config.save_every,
            self.config.save_every_unit,
            train_progress,
            start_at_zero=False,
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate",
            self.config.validate_after,
            self.config.validate_after_unit,
            train_progress,
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step",
            self.config.gradient_accumulation_steps,
            TimeUnit.STEP,
            train_progress,
            start_at_zero=False,
        )

    def __apply_fused_back_pass(self, scaler):
        if self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
            if self.config.gradient_accumulation_steps > 1:
                print(
                    "Warning: activating fused_back_pass with gradient_accumulation_steps > 1 does not reduce VRAM usage."
                )

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

    def _handle_pause_logic(self):
        """Executa a lógica de pausa, movendo o modelo e esperando."""
        if not self.is_paused:  # Segurança extra
            return

        logFun("Iniciando Pausa...", lvl="LOOP")
        self.callbacks.on_update_status("Pausing... Moving model to CPU")
        try:
            self.model.to(self.temp_device)  # Mover para CPU
            self.model.eval()  # Garantir modo eval
            torch_gc()  # Limpar VRAM
            logFun(f"Modelo movido para {self.temp_device}. VRAM liberada.", lvl="success")
            self.callbacks.on_update_status(f"Paused. Model on {self.temp_device}. Toggle switch to resume.")
            # Notificar UI que a pausa iniciou e o switch pode ser reativado (para desligar)
            if hasattr(self.callbacks, "on_pause_initiated"):
                self.callbacks.on_pause_initiated()

            # Loop de espera pela retomada
            while self.is_paused:
                if self.commands.get_stop_command():
                    logFun("Comando STOP recebido durante a pausa. Interrompendo.", lvl="warning")
                    self.is_paused = False  # Força a saída do loop de pausa
                    # Mantém o comando de stop ativo para o loop principal
                    break

                if self.commands.get_and_reset_resume_request():
                    logFun("Comando RESUME recebido.", lvl="info")
                    self.is_paused = False  # Sinaliza para sair do loop
                    self.pause_request_locked = False  # Desbloqueia a UI
                    # Notificar UI que o resume começou (switch ainda ativo)
                    if hasattr(self.callbacks, "on_resume_started"):
                        self.callbacks.on_resume_started()
                    break  # Sai do loop de espera

                time.sleep(0.5)  # Evita busy-waiting, checa a cada 0.5s

            if not self.commands.get_stop_command():  # Só retoma se não for parar
                logFun("Retomando treinamento...", lvl="info")
                self.callbacks.on_update_status("Resuming... Moving model to GPU")
                try:
                    # Recarregar para o dispositivo de treino
                    self.model_setup.setup_train_device(self.model, self.config)
                    torch_gc()  # Limpeza extra
                    logFun(f"Modelo movido de volta para {self.config.train_device}.", lvl="success")
                    self.callbacks.on_update_status("Training resumed.")
                    # Notificar UI que o resume foi concluído
                    if hasattr(self.callbacks, "on_resume_completed"):
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
            self.is_paused = False  # Garante que não fique preso no estado pausado
            self.pause_request_locked = False
            # Considerar parar o treino em caso de erro grave aqui
            self.commands.stop()
            self.callbacks.on_update_status(f"Error during pause/resume: {e}")

    def latent_ssim(self, pred_lat: torch.Tensor, tgt_lat: torch.Tensor) -> torch.Tensor:
        """
        SSIM proxy p/ latentes 128×128 (ou menor).
        - Se H < 128: faz upscale NN → 128.
        - Normaliza ambos tensores para [0,1] antes do SSIM.
        - Usa janela 11×11 (ou a maior ímpar que couber).
        - Calcula SSIM em fp32 e devolve no dtype original.
        """
        if pred_lat.shape[-1] < 128:
            pred_lat = F.interpolate(pred_lat, size=128, mode="nearest")
            tgt_lat = F.interpolate(tgt_lat, size=128, mode="nearest")

        with torch.no_grad():
            # 1) normalização conjunta → [0,1]
            stacked = torch.cat([pred_lat, tgt_lat], dim=0)
            min_val = stacked.min()
            max_val = stacked.max()
            data_rng = (max_val - min_val).clamp(min=1e-7)  # evita div/0
            pred_norm = (pred_lat.float() - min_val) / data_rng
            tgt_norm = (tgt_lat.float() - min_val) / data_rng

            ssim32 = ssim(pred_norm, tgt_norm, data_range=1.0, size_average=True, win_size=11)

        return ssim32.to(dtype=pred_lat.dtype)

    def train(self):
        scheduler_step_counter = 0
        discard_counter = 0
        train_device = torch.device(self.config.train_device)
        train_progress = self.model.train_progress

        # Determine target device and dtype from a model parameter if available
        def wrap_scheduler_step(orig_step):
            def wrapped(*args, **kwargs):
                nonlocal scheduler_step_counter
                scheduler_step_counter += 1
                logFun(f"[DEBUG] scheduler.step() chamado {scheduler_step_counter} vezes", lvl="DEBUG")
                logFun(f"[DEBUG] scheduler.last_epoch = {lr_scheduler.last_epoch}", lvl="DEBUG")
                return orig_step(*args, **kwargs)

            return wrapped

        def prepare_mask(mask: torch.Tensor, ref: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
            """Binariza + broadcasta máscara para ter shape/dtype de `ref`."""
            m = (mask > thresh).to(dtype=ref.dtype, device=ref.device)
            if m.ndim < ref.ndim:  # [B,H,W] → [B,1,H,W]
                m = m.unsqueeze(1)
            if m.shape[1] == 1 and ref.shape[1] != 1:
                m = m.expand(ref.shape[0], ref.shape[1], *m.shape[2:])
            return m

        # Verificar se é só cache (mantém como está)
        if self.config.only_cache:
            self._handle_cache_only_mode()
            return

        if self.config.only_cache:
            self.callbacks.on_update_status("caching")

            with ProgressContext("Caching latents", self.config.epochs - train_progress.epoch) as progress:
                for _epoch in range(train_progress.epoch, self.config.epochs, 1):
                    self.data_loader.get_data_set().start_next_epoch()
                    progress.update(1)
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None
        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False
        accumulated_loss = 0.0
        ema_loss = 0.0
        lr_scheduler = None

        # Inicia o sistema de display principal (que agora gerencia o Live para tudo)
        self._setup_training_display()
        self._total_training_time_start_s = time.monotonic()

        try:
            for _epoch in range(train_progress.epoch, self.config.epochs, 1):
                self._epoch_start_time_s = time.monotonic()

                if self.is_paused:
                    logFun(
                        f"Treino iniciado em estado PAUSADO (Epoch {train_progress.epoch}). Aguardando resume...",
                        lvl="info",
                    )
                    self._handle_pause_logic()
                    if self.commands.get_stop_command():  # Se o stop foi dado durante a pausa inicial
                        logFun("Comando STOP ativo após pausa inicial. Encerrando.", lvl="warning")
                        break  # Sai do loop de épocas

                self.callbacks.on_update_status("training")

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
                        global_step=train_progress.global_step,
                    )

                gps_instance: TrainGPS | None = getattr(self.model, "deltas", None)
                current_epoch_length = self.data_loader.get_data_set().approximate_length()

                for batch_idx, batch in enumerate(self.data_loader.get_data_loader()):
                    step_start_time_s = time.monotonic()
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
                            self.backup(train_progress, True, lambda msg: logFun(msg, lvl="LOOP"))
                            transferred_to_temp_device = True

                        if self.commands.get_and_reset_save_command():
                            self.model.to(self.temp_device)
                            self.save(train_progress, True, lambda msg: logFun(msg, lvl="LOOP"))
                            transferred_to_temp_device = True

                        if transferred_to_temp_device:
                            self.model_setup.setup_train_device(self.model, self.config)

                    with TorchMemoryRecorder(enabled=False):
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                        t_used = getattr(self.model_setup, "current_timestep", None)
                        predicted_tensor_from_model = model_output_data[
                            "predicted"
                        ]  # Saída bruta do modelo (e.g., bf16)

                        mask_fp32 = None
                        if self.config.masked_training:
                            mask_bf16 = prepare_mask(batch["latent_mask"], predicted_tensor_from_model)
                            mask_fp32 = mask_bf16.to(torch.float32)
                            predicted_tensor_from_model.mul_(mask_bf16)
                            model_output_data["target"].mul_(mask_bf16)

                            def _grad_mask(g):
                                g.mul_(mask_fp32)
                                return g

                            hook_handle = predicted_tensor_from_model.register_hook(_grad_mask)
                        else:
                            hook_handle = None

                        loss = self.model_setup.calculate_loss(
                            self.model,
                            batch,
                            model_output_data,
                            self.config,
                            train_progress,
                        )

                        loss = loss / self.config.gradient_accumulation_steps

                        if self.config.debugoi:
                            predicted_tensor_from_model.retain_grad()

                        # START update_timestep_perf
                        if t_used is not None and train_progress.global_step >= 0:
                            # no batch=1, t_used pode ser um tensor de shape (1,), então faz:
                            t_idx = t_used.item() if isinstance(t_used, Tensor) else int(t_used)
                            l_scalar = loss.item()
                            # EMA de loss por timestep
                            prev = float(self.timestep_perf[t_idx])
                            self.timestep_perf[t_idx] = self.ema_alpha * prev + (1 - self.ema_alpha) * l_scalar
                            self.timestep_count[t_idx] += 1
                        # END update_timestep_perf

                        if scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        if hook_handle is not None:
                            hook_handle.remove()

                        if self.config.debugoi:
                            if predicted_tensor_from_model.grad is not None:
                                leak = (predicted_tensor_from_model.grad * (1 - mask_fp32)).abs().max()
                                logFun(f"Leak grad (bf16→FP32) = {leak.item():.2e}")
                            else:
                                logFun("WARNING: grad é None — verifique se retain_grad foi chamado antes do backward")

                        has_gradient = True
                        accumulated_loss += loss.item()
                        if self.__is_update_step(train_progress):
                            if (
                                scaler
                                and self.config.optimizer.optimizer.supports_fused_back_pass()
                                and self.config.optimizer.fused_back_pass
                            ):
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

                            # Scheduler de learning rate
                            lr_scheduler.step()

                            # Reset de gradientes
                            self.model.optimizer.zero_grad(set_to_none=True)
                            has_gradient = False

                            self._current_step_duration_s = time.monotonic() - step_start_time_s
                            # EMA step time
                            if self._ema_step_time_s is None:
                                self._ema_step_time_s = self._current_step_duration_s
                            else:
                                self._ema_step_time_s = (
                                    self._ema_alpha * self._current_step_duration_s
                                    + (1 - self._ema_alpha) * self._ema_step_time_s
                                )

                            self._epoch_time_elapsed_s = time.monotonic() - self._epoch_start_time_s

                            if self._ema_step_time_s is not None:
                                self._avg_step_time_epoch_s = self._ema_step_time_s

                            # Atualiza o display do Rich
                            self._update_training_display(current_loss=accumulated_loss, current_ema_loss=ema_loss)

                            self.model_setup.report_to_tensorboard(self.model, self.config, lr_scheduler)

                            self.tensorboard.add_scalar(
                                "loss/train_step",
                                accumulated_loss,
                                train_progress.global_step,
                            )
                            ema_loss = ema_loss or accumulated_loss
                            ema_loss = (ema_loss * 0.99) + (accumulated_loss * 0.01)
                            self.tensorboard.add_scalar(
                                "smooth_loss/train_step",
                                ema_loss,
                                train_progress.global_step,
                            )

                            accumulated_loss = 0.0
                            if self.model.ema:
                                update_step = (train_progress.global_step, self.config.gradient_accumulation_steps)
                                self.tensorboard.add_scalar(
                                    "ema_decay",
                                    self.model.ema.get_current_decay(update_step),
                                    train_progress.global_step,
                                )
                                self.model.ema.step(self.parameters, update_step)
                            self.one_step_trained = True

                    if self.config.validation:
                        self.__validate(train_progress)

                    train_progress.next_step(self.config.batch_size)
                    self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                # START adjust_timestep_counts
                # só atualiza após burn-in (pelo menos uma época completa)
                if train_progress.global_step >= self.burn_in_steps:
                    num_timesteps = 1000
                    steps_per_epoch = self._steps_per_epoch  # já configurado no start
                    # 1) base_counts: distribuição estratificada pura
                    base = steps_per_epoch // num_timesteps
                    rem = steps_per_epoch % num_timesteps
                    base_counts = torch.full((num_timesteps,), base, dtype=torch.long, device=self.config.train_device)
                    if rem > 0:
                        base_counts[:rem] += 1

                    # 2) calcular adj_factor a partir de perf (norm e clamp)
                    perf = self.timestep_perf.clone()
                    # evita dividir por zero
                    mn, mx = float(perf.min()), float(perf.max())
                    perf_norm = (perf - mn) / (mx - mn + 1e-8)
                    scale = getattr(self.config, "timestep_adj_scale", 0.5)
                    adj_factor = (1.0 + (perf_norm - 0.5) * scale).clamp(0.7, 1.3)

                    # 3) alloc = base_counts * adj_factor; normaliza para steps_per_epoch
                    alloc = (base_counts.float() * adj_factor).round().to(dtype=torch.long)
                    total_alloc = int(alloc.sum().item())
                    # se sobrar, diminui dos maiores
                    while total_alloc > steps_per_epoch:
                        idx_max = int(torch.argmax(alloc))
                        alloc[idx_max] -= 1
                        total_alloc -= 1
                    # se faltar, distribui +1 para timesteps com maior adj_factor
                    while total_alloc < steps_per_epoch:
                        idx_max = int(torch.argmax(adj_factor))
                        alloc[idx_max] += 1
                        total_alloc += 1

                    # 4) finalmente passa esta alloc para o mixin
                    self.model_setup.set_epoch_timestep_alloc(alloc)

                train_progress.next_epoch()  # Avança a epoch para a próxima iteração do loop externo

                # Log de final de epoch no console Rich
                final_epoch_duration = time.monotonic() - self._epoch_start_time_s
                avg_step_final_epoch = self._avg_step_time_epoch_s  # Usa o valor final calculado
                self.console.log(
                    f"[bold green]Epoch {train_progress.epoch} concluída em {format_time_delta(final_epoch_duration)} "
                    f"(Avg step: {avg_step_final_epoch:.3f}s/it)[/bold green]"
                )

                if self.commands.get_and_reset_pause_request():
                    logFun(
                        f"Requisição de PAUSA recebida. Será executada ao final da Epoch {train_progress.epoch - 1}.",
                        lvl="info",
                    )
                    self.pause_requested_at_epoch_end = True
                    self.pause_request_locked = True  # Trava a UI
                    # Notificar a UI que a requisição foi aceita e o switch está travado
                    if hasattr(self.callbacks, "on_pause_request_accepted"):
                        self.callbacks.on_pause_request_accepted()

                # 2. Executar a pausa se foi agendada
                if self.pause_requested_at_epoch_end and not self.is_paused:
                    self.is_paused = True  # Marca como pausado
                    self.pause_requested_at_epoch_end = False  # Limpa a flag de agendamento
                    # A trava (pause_request_locked) continua TRUE até o resume

                    # Chama a função que move o modelo e entra no loop de espera
                    self._handle_pause_logic()

                # Checagem de STOP ao final da época
                if self.commands.get_stop_command():
                    logFun("Comando STOP ativo no final da época. Encerrando...", lvl="info")
                    break  # Sai do loop de épocas

                # 1. TrainGPS salva os deltas da epoch
                if gps_instance is not None:
                    if self.config.train_gps_save_it:
                        try:
                            epoch_idx = train_progress.epoch - 1
                            gps_instance.log_group_deltas(epoch_idx)
                        except Exception as e:
                            logFun(
                                f"[TrainGPS] Erro ao logar deltas do grupo na época {train_progress.epoch - 1}: {e}",
                                lvl="error",
                            )
                            traceback.print_exc()

                if self.commands.get_stop_command():
                    return
        finally:
            self._stop_training_display()  # Para o Live display principal
            cleanup_global_progress()  # Limpa o objeto Progress global (do logFun)

        # Exibir summary final
        total_training_duration = (
            time.monotonic() - self._total_training_time_start_s if self._total_training_time_start_s else 0
        )
        logFun(f"Treinamento completo! Tempo total: {format_time_delta(total_training_duration)}", lvl="SUCCESS")

    def end(self):
        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{self.model.train_progress.filename_string()}{self.config.output_model_format.file_extension()}",
        )
        if self.is_paused:
            logFun(
                "Finalizando treinamento enquanto estava pausado. Tentando retomar brevemente para salvar.",
                lvl="warning",
            )
            # Força a saída da pausa (sem esperar comando) e tenta mover para GPU para salvar
            self.is_paused = False
            self.pause_request_locked = False
            try:
                # Tenta mover de volta pra GPU rapidamente
                self.model_setup.setup_train_device(self.model, self.config)
                torch_gc()
                logFun("Modelo movido para GPU para salvamento final.", lvl="info")
            except Exception as e:
                logFun(
                    f"Falha ao mover modelo para GPU no final (estava pausado): {e}. Salvando do CPU ({self.temp_device}).",
                    lvl="error",
                    _console=self.console,
                )  # Usar _console aqui
                # O modelo já está no self.temp_device, o save deve funcionar

        if self.one_step_trained:
            self.model.to(self.temp_device)
            torch_gc()

            if self.config.backup_before_save:
                self.backup(self.model.train_progress)  # Backup já usa o modelo no temp_device

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            self.callbacks.on_update_status("saving the final model")

            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=False)
            if os.path.isdir(self.config.output_model_destination) and self.config.output_model_format.is_single_file():
                save_path = os.path.join(
                    self.config.output_model_destination,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}",
                )
            else:
                save_path = self.config.output_model_destination
            logFun("Saving " + save_path, lvl="LOOP")  # Usar logFun aqui

            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype(),
            )
        elif self.model is not None:
            self.model.to(self.temp_device)

        model_filename = os.path.basename(save_path)
        model_name, _ = os.path.splitext(model_filename)  # Get model name without extension

        # --- Save Delta Pattern (If train_gps_save_it is True AND instance exists) ---
        gps_instance: TrainGPS | None = getattr(self.model, "deltas", None)
        # A condição agora é apenas checar a flag de salvar e se o módulo foi inicializado
        if getattr(self.config, "train_gps_save_it", False) and gps_instance is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Incluir o número da Run no nome do arquivo para clareza
                output_dir = os.path.join(self.config.workspace_dir, "training_deltas")
                os.makedirs(output_dir, exist_ok=True)
                delta_filename = f"{model_name}_Deltas_Run{self.run_number}_{timestamp}.json"
                delta_save_path = os.path.join(output_dir, delta_filename)

                logFun(f"[TrainGPS] Salvando deltas (Run {self.run_number}) em: {delta_save_path}", lvl="info")
                # A função save_group_deltas salva o estado atual do delta_log_by_module
                # que foi acumulado durante esta run específica.
                gps_instance.save_group_deltas(delta_save_path)
            except Exception as e:
                logFun(f"[TrainGPS] Erro ao salvar deltas (Run {self.run_number}): {e}", lvl="error")
                traceback.print_exc()

        if self.recorder:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use a consistent naming scheme/directory if desired
                output_dir = os.path.join(self.config.workspace_dir, "data_recorder")  # Or reuse training_deltas dir?
                os.makedirs(output_dir, exist_ok=True)
                # Include Run number for clarity
                profile_filename = f"{model_name}_Profile_Run{self.run_number}_{timestamp}.json.gz"
                profile_save_path = os.path.join(output_dir, profile_filename)

                logFun(f"[DataRecorder] Salvando perfil (Run {self.run_number}) em: {profile_save_path}", lvl="info")

                # DataRecorder's dump method now only takes the path
                # It saves d_hat_final and d_coef_base internally collected.
                self.recorder.dump(profile_save_path)

            except Exception as e:
                logFun(f"[DataRecorder] Erro ao salvar perfil de dinâmica (Run {self.run_number}): {e}", lvl="error")
                traceback.print_exc()

        self.tensorboard.close()

        if self.config.tensorboard:
            super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
        torch_gc()
