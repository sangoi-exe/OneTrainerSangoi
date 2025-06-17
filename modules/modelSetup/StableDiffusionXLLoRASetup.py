import torch

from collections import defaultdict
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import re
import itertools
from torch import nn
from datetime import datetime
from pathlib import Path

PRESETS = {
    "attn-mlp": ["attentions"],
    "attn-only": ["attn"],
    "full": [],
}


class StableDiffusionXLLoRASetup(
    BaseStableDiffusionXLSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    @staticmethod
    def _register_group(
            prefix: str,
            params,
            collection: NamedParameterGroupCollection,
            lr: float,
    ):
        collection.add_group(NamedParameterGroup(
            unique_name=prefix,
            display_name=prefix,
            parameters=params,
            learning_rate=lr,
        ))

    @staticmethod
    def _classify_unet_param(full_name: str, strategy: str) -> str:
        """
        Decide em qual bucket o módulo cai conforme `strategy`.
        """
        stage, block = "other", "other"
        # Detecta o estágio principal e o índice do bloco (down, up ou mid)
        if m := re.search(r"down[_\.]blocks[_\.](\d+)", full_name):
            stage, block = "down", m.group(1)
        elif m := re.search(r"up[_\.]blocks[_\.](\d+)", full_name):
            stage, block = "up", m.group(1)
        elif "mid_block" in full_name:
            stage, block = "mid", "mid"

        # refino do identificador de bloco
        # Se existir um sub-bloco (attentions, resnets, up|down-samplers, transformer_blocks),
        # anexamos a primeira letra + índice ao id do bloco.
        #   Ex.  down_blocks.1.attentions.0.* → block = "1a0"
        #        down_blocks.2.resnets.3.*    → block = "2r3"
        #        up_blocks.0.upsamplers.0.*   → block = "0u0"
        #        mid_block.attentions.0.*     → block = "mida0"
        inner = re.search(r"(attentions|resnets|upsamplers|downsamplers|transformer_blocks)[_\.](\d+)", full_name)
        if inner:
            kind, idx = inner.groups()
            block = f"{block}{kind[0]}{idx}"

        _type = "misc"
        name = full_name.replace('.', '_')
        tokens = name.split('_')
        joined  = '_'.join(tokens)

        def has(substr: str) -> bool: 
            return substr in joined

        if has("attn1"): _type = "attn1"
        elif has("attn2"):  _type = "attn2"
        elif has("ff_net"): _type = "ff"
        elif joined.endswith("proj_in"):   _type = "proj_in"
        elif joined.endswith("proj_out"):  _type = "proj_out"
        elif m := re.search(r"_to_([kqv])$", joined):
            _type = f"to_{m.group(1)}"
        elif "conv_shortcut" in joined:  _type = "conv_shortcut"
        elif re.search(r"conv\d?$", joined): _type = "conv"
        elif "time_emb_proj" in joined: _type = "time_emb_proj"
        elif "linear" in joined: _type = "linear"
        else:
            tail = re.sub(r"\d+$", "", tokens[-1])
            _type = tail or "misc"

        match strategy:
            case "module":      return full_name
            case "block":       return f"{stage}{block}"
            case "stage":       return stage
            case "type":        return _type
            case "stage_type":  return f"{stage}_{_type}"
            case "block_type":  return f"{stage}{block}_{_type}"
            case _:             return full_name
            

    @staticmethod
    def _add_lora_param_groups_te(wrapper: LoRAModuleWrapper, prefix: str, collection: NamedParameterGroupCollection, base_lr: float):
        """
        Cria um NamedParameterGroup **por** sub-módulo LoRA dentro do wrapper.

        prefix: identifica qual parte do modelo (unet, te1, te2…).
        """
        for name, lora_mod in wrapper.lora_modules.items():
            unique = f"{prefix}.{name}"
            collection.add_group(NamedParameterGroup(
                unique_name=unique,
                display_name=unique,
                parameters=lora_mod.parameters(),
                learning_rate=base_lr,
            ))

    def _add_lora_param_groups(
            self,
            wrapper: LoRAModuleWrapper,
            prefix: str,
            collection: NamedParameterGroupCollection,
            base_lr: float,
            strategy: str = "module",
    ):
        """
        Agrupa parâmetros LoRA segundo `strategy`.
        """
        # Agora guardamos (nome, módulo) para poder listar depois
        buckets: defaultdict[str, list[tuple[str, nn.Module]]] = defaultdict(list)
        for name, lora_mod in wrapper.lora_modules.items():
            key = self._classify_unet_param(name, strategy)
            buckets[key].append((name, lora_mod))

        for key, items in buckets.items():
            if not items:
                print(f"[WARN] bucket '{key}' sem parâmetros, ignorado.")
                continue
            params_iter = (m.parameters() for (_, m) in items)
            self._register_group(
                prefix=f"{prefix}.{key}",
                params=itertools.chain.from_iterable(params_iter),
                collection=collection,
                lr=base_lr,
            )

        self._dump_bucket_debug(prefix, strategy, buckets)
        print(f"[INFO] {prefix}: {len(buckets)} grupos criados via strategy='{strategy}'.")

    @staticmethod
    def _dump_bucket_debug(prefix: str, strategy: str, buckets: dict[str, list[tuple[str, nn.Module]]]):
        """Salva um TXT com timestamp listando bucket -> módulos -> n_params."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("./debug")
        out_dir.mkdir(exist_ok=True)
        filename = out_dir / f"{prefix}_{strategy}_{ts}.txt"

        lines: list[str] = []
        for key, items in sorted(buckets.items()):
            total = sum(p.numel() for _, m in items for p in m.parameters())
            lines.append(f"[{key}] total_params={total:,}, num_modules={len(items)}")
            for mod_name, mod in items:
                mod_params = sum(p.numel() for p in mod.parameters())
                lines.append(f"    {mod_name:<60}\t{mod_params:,}")
            lines.append("")  # linha em branco separando buckets

        filename.write_text("\n".join(lines))
        print(f"[DEBUG] Buckets dump salvo em: {filename}")

    def create_parameters(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        params_by_module = getattr(config, "parameters_by_module", False)

        if not params_by_module:
            if config.text_encoder.train:
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name="text_encoder_1_lora",
                    parameters=model.text_encoder_1_lora.parameters(),
                    learning_rate=config.text_encoder.learning_rate,
                ))
                
            if config.text_encoder_2.train:
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name="text_encoder_2_lora",
                    parameters=model.text_encoder_2_lora.parameters(),
                    learning_rate=config.text_encoder_2.learning_rate,
                ))

            if config.unet.train:
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name="unet_lora",
                    parameters=model.unet_lora.parameters(),
                    learning_rate=config.unet.learning_rate,
                ))
        else:
            if config.text_encoder.train:
                self._add_lora_param_groups_te(
                    model.text_encoder_1_lora,
                    "text_encoder_1_lora", parameter_group_collection,
                    config.text_encoder.learning_rate
                )
                
            if config.text_encoder_2.train:
                self._add_lora_param_groups_te(
                    model.text_encoder_2_lora,
                    "text_encoder_2_lora", parameter_group_collection,
                    config.text_encoder_2.learning_rate
                )

            if config.unet.train:
                self._add_lora_param_groups(
                    model.unet_lora,
                    "unet_lora",
                    parameter_group_collection,
                    config.unet.learning_rate,
                    strategy=getattr(config, "param_group_strategy", "module"),
                )

        if config.train_any_embedding():
            if config.text_encoder.train_embedding:
                self._add_embedding_param_groups(
                    model.embedding_wrapper_1, parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_1"
                )

            if config.text_encoder_2.train_embedding:
                self._add_embedding_param_groups(
                    model.embedding_wrapper_2, parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_2"
                )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_1_lora is not None:
            train_text_encoder_1 = config.text_encoder.train and \
                                   not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_1_lora.requires_grad_(train_text_encoder_1)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]

            train_embedding_1 = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_1_vector.requires_grad_(train_embedding_1)

            train_embedding_2 = \
                embedding_config.train \
                and config.text_encoder_2.train_embedding \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_2_vector.requires_grad_(train_embedding_2)

        if model.text_encoder_2_lora is not None:
            train_text_encoder_2 = config.text_encoder_2.train and \
                                   not self.stop_text_encoder_2_training_elapsed(config, model.train_progress)
            model.text_encoder_2_lora.requires_grad_(train_text_encoder_2)

        if model.unet_lora is not None:
            train_unet = config.unet.train and \
                         not self.stop_unet_training_elapsed(config, model.train_progress)
            model.unet_lora.requires_grad_(train_unet)

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        create_te1 = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te1")
        create_te2 = config.text_encoder_2.train or state_dict_has_prefix(model.lora_state_dict, "lora_te2")

        model.text_encoder_1_lora = LoRAModuleWrapper(
            model.text_encoder_1, "lora_te1", config
        ) if create_te1 else None

        model.text_encoder_2_lora = LoRAModuleWrapper(
            model.text_encoder_2, "lora_te2", config
        ) if create_te2 else None

        model.unet_lora = LoRAModuleWrapper(
            model.unet, "lora_unet", config, config.lora_layers.split(",")
        )

        if model.lora_state_dict:
            if create_te1:
                model.text_encoder_1_lora.load_state_dict(model.lora_state_dict)
            if create_te2:
                model.text_encoder_2_lora.load_state_dict(model.lora_state_dict)

            model.unet_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if config.text_encoder.train:
            model.text_encoder_1_lora.set_dropout(config.dropout_probability)
        if config.text_encoder_2.train:
            model.text_encoder_2_lora.set_dropout(config.dropout_probability)
        model.unet_lora.set_dropout(config.dropout_probability)

        if create_te1:
            model.text_encoder_1_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_1_lora.hook_to_module()
        if create_te2:
            model.text_encoder_2_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_2_lora.hook_to_module()

        model.unet_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.unet_lora.hook_to_module()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        vae_on_train_device = config.align_prop or not config.latent_caching
        text_encoder_1_on_train_device = \
            config.train_text_encoder_or_embedding()\
            or config.align_prop \
            or not config.latent_caching
        text_encoder_2_on_train_device = \
            config.train_text_encoder_2_or_embedding() \
            or config.align_prop \
            or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        if config.text_encoder.train:
            model.text_encoder_1.train()
        else:
            model.text_encoder_1.eval()

        if config.text_encoder_2.train:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.eval()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            model.embedding_wrapper_1.normalize_embeddings()
            model.embedding_wrapper_2.normalize_embeddings()
        self.__setup_requires_grad(model, config)
