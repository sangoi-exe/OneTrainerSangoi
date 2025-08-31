# SDXL Training Pipeline Code Catalog

Lista de arquivos ligados à pipeline de treino do Stable Diffusion XL (SDXL) neste repositório. Os caminhos são relativos à raiz do projeto.

## Configurações e Dependências
- `README.md` – menciona suporte ao SDXL.
- `requirements-global.txt` – inclui dependência `invisible-watermark` necessária para a pipeline SDXL.

## Especificações de Modelo
- `resources/sd_model_spec/sd_xl_base_1.0.json` – especificação do modelo base SDXL.
- `resources/sd_model_spec/sd_xl_base_1.0-embedding.json` – especificação para treino de embeddings.
- `resources/sd_model_spec/sd_xl_base_1.0-lora.json` – especificação para treino de LoRA.
- `resources/sd_model_spec/sd_xl_base_1.0_inpainting.json` – especificação do modelo base para inpainting.
- `resources/sd_model_spec/sd_xl_base_1.0_inpainting-lora.json` – especificação LoRA para inpainting.
- `resources/sd_model_spec/sd_xl_base_1.0_Inpainting-embedding.json` – especificação de embeddings para inpainting.

## Módulos Principais
- `modules/modelSampler/StableDiffusionXLSampler.py` – sampler especializado para SDXL.
- `modules/modelSetup/BaseStableDiffusionXLSetup.py` – lógica base de setup para SDXL.
- `modules/modelSetup/mixin/ModelSetupDebugMixin.py` – utilidades de depuração com projeção latente SDXL.
- `modules/modelSaver/mixin/DtypeModelSaverMixin.py` – salva modelos marcando versão base "sdxl_".
- `modules/modelSaver/stableDiffusionXL/StableDiffusionXLModelSaver.py` – salva modelos SDXL convertendo de Diffusers para ckpt.
- `modules/modelLoader/StableDiffusionXLLoRAModelLoader.py` – carregador de LoRAs SDXL.
- `modules/modelLoader/StableDiffusionXLFineTuneModelLoader.py` – carregador de modelos fine‑tuned SDXL.
- `modules/modelLoader/StableDiffusionXLEmbeddingModelLoader.py` – carregador de embeddings SDXL.
- `modules/modelLoader/stableDiffusionXL/StableDiffusionXLModelLoader.py` – carregador de modelos base SDXL.
- `modules/util/convert/convert_sdxl_diffusers_to_ckpt.py` – conversão de modelos SDXL Diffusers para checkpoint.

## Scripts e Utilitários Diversos
- `modules/sangoi/xiforimpola.py` – mescla text encoder SDXL com ViT‑BigG.
- `modules/sangoi/CrossAttnMapsAnalyzer.py` – analisa mapas de atenção e projeta latentes SDXL.
- `modules/sangoi/readTest.py` – script de leitura de pesos SDXL.
- `modules/ui/LoraTab.py` – UI com presets de treino LoRA para SDXL.

## Exemplos e Documentação do Diffusers
- `src/diffusers/examples/dreambooth/README_sdxl.md` – instruções para Dreambooth com SDXL.
- `src/diffusers/examples/controlnet/train_controlnet_sdxl.py` – script de treino ControlNet para SDXL.
- `src/diffusers/examples/t2i_adapter/train_t2i_adapter_sdxl.py` – script de treino T2I‑Adapter para SDXL.
- `src/diffusers/examples/research_projects/controlnet/train_controlnet_webdataset.py` – treino de ControlNet SDXL com webdataset.
- `src/diffusers/examples/research_projects/promptdiffusion/convert_original_promptdiffusion_to_diffusers.py` – conversão de PromptDiffusion para Diffusers (SDXL).
- `src/diffusers/docs/source/en/using-diffusers/sdxl.md` – guia de uso do SDXL.
- `src/diffusers/docs/source/en/using-diffusers/sdxl_turbo.md` – guia de uso do SDXL Turbo.
- `src/diffusers/docs/source/ko/api/pipelines/stable_diffusion/stable_diffusion_xl.md` – documentação da API SDXL.
- `src/diffusers/docs/source/ko/using-diffusers/sdxl_turbo.md` – guia em coreano do SDXL Turbo.
- `src/diffusers/docs/source/en/optimization/onnx.md` – inclui otimizações ONNX para SDXL.
- `src/diffusers/docs/source/en/using-diffusers/other-formats.md` – conversões de formatos envolvendo SDXL.
- `src/diffusers/docs/source/en/using-diffusers/weighted_prompts.md` – uso de prompts ponderados com SDXL.

## Pipelines e Scripts do Diffusers
- `src/diffusers/scripts/convert_stable_diffusion_controlnet_to_onnx.py` – conversão de ControlNet SDXL para ONNX.
- `src/diffusers/scripts/convert_stable_diffusion_controlnet_to_tensorrt.py` – conversão de ControlNet SDXL para TensorRT.
- `src/diffusers/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py` – conversão de checkpoints SDXL para Diffusers.
- `src/diffusers/src/diffusers/pipelines/pag/__init__.py` – registro de pipelines PAG incluindo variantes SDXL.
- `src/diffusers/src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_xl.py` – pipeline PAG ControlNet SDXL.
- `src/diffusers/src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_xl_img2img.py` – pipeline PAG ControlNet SDXL img2img.
- `src/diffusers/src/diffusers/pipelines/controlnet/__init__.py` – expõe pipeline ControlNet SDXL.
- `src/diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py` – pipeline ControlNet SDXL img2img.
- `src/diffusers/src/diffusers/pipelines/controlnet_xs/__init__.py` – inclui variantes ControlNet XS para SDXL.

## Testes do Diffusers
- `src/diffusers/tests/single_file/test_stable_diffusion_xl_adapter_single_file.py` – testes de adapter SDXL.
- `src/diffusers/tests/single_file/test_stable_diffusion_xl_single_file.py` – testes do pipeline SDXL.
- `src/diffusers/tests/single_file/test_stable_diffusion_xl_img2img_single_file.py` – testes de img2img SDXL.
- `src/diffusers/tests/single_file/test_stable_diffusion_xl_controlnet_single_file.py` – testes de ControlNet SDXL.
- `src/diffusers/tests/lora/test_lora_layers_sdxl.py` – testes de camadas LoRA para SDXL.

## Fluxo de Treino LoRA/DoRA – Funções Principais

- `modules/trainer/BaseTrainer.py` – interface base de treinamento.
  - `__init__(config, callbacks, commands)` – define dispositivos e objetos centrais.
  - `start()`, `train()`, `end()`, `backup(train_progress)` – ganchos abstratos.
  - `create_model_loader()`, `create_model_setup()`, `create_data_loader()`,
    `create_model_saver()`, `create_model_sampler()` – fábricas de componentes.
- `modules/trainer/GenericTrainer.py` – loop genérico de treinamento.
  - `_handle_pause_logic()` – move o modelo para CPU e aguarda retomada.
  - `start()` – carrega modelo, dataloaders e dispara o treino.
  - `train()` – laço principal com gradiente acumulado, hooks de gradiente e amostragens.
  - `backup()` e `end()` – salvam checkpoints e finalizam o processo.
- `modules/modelLoader/StableDiffusionXLLoRAModelLoader.py`
  - `_default_model_spec_name(model_type)` – escolhe o spec padrão.
  - `load(model_type, model_names, weight_dtypes)` – carrega modelo base, LoRA e embeddings.
- `modules/model/StableDiffusionXLModel.py`
  - Funções de movimentação: `vae_to`, `text_encoder_to`, `unet_to`, `to` e `eval`.
  - `create_pipeline()` – monta a pipeline do diffusers.
  - `encode_text(...)` – tokeniza e codifica texto com dropout opcional.
- `modules/module/LoRAModule.py`
  - `PeftBase` com `hook_to_module()` e `make_weight()` para injeção de pesos.
  - `LoRAModule` e `DoRAModule` – implementações das camadas adaptativas.
  - `LoRAModuleWrapper` – cria, move e coleta os módulos LoRA/DoRA.
- `modules/modelSetup/BaseStableDiffusionXLSetup.py`
  - `setup_optimizations()` – aplica otimizações de memória/atenção.
  - `_setup_additional_embeddings()` e `_setup_embedding_wrapper()` – configuram embeddings extras.
  - `predict()` – gera ruído/latente e dados auxiliares do modelo.
  - `calculate_loss()` – delega cálculo de loss para o mixin correspondente.
- `modules/modelSetup/StableDiffusionXLLoRASetup.py`
  - `_register_group()` e `_classify_unet_param()` – agrupam parâmetros LoRA.
  - `create_parameters()` – monta grupos de parâmetros para o otimizador.
  - `setup_model()` – cria wrappers LoRA para UNet e text encoders.
  - `setup_train_device()` e `after_optimizer_step()` – controlam dispositivos e gradientes.
- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` – mixin de cálculo de perdas.
  - `__log_cosh_loss()`, `__masked_losses()`, `__unmasked_losses()`.
  - Funções de ponderação: `__min_snr_weight()`, `__p2_loss_weight()`, `__sigma_loss_weight()`,
    `__sangoi_loss_weighting()`.
  - `_diffusion_losses()` e `_flow_matching_losses()` – retornam o tensor de loss final.
- `modules/modelSetup/mixin/ModelSetupNoiseMixin.py`
  - `_create_noise()` – gera ruído a partir do scheduler.
  - `_get_timestep_discrete()` / `_get_timestep_continuous()` – convertem timesteps.
- `modules/modelSetup/mixin/ModelSetupDiffusionMixin.py`
  - `_add_noise_discrete()` e `_add_noise_continuous()` – aplicam ruído ao latente.
- `modules/modelSetup/mixin/ModelSetupEmbeddingMixin.py`
  - `_create_new_embedding()` e `_add_embedding_to_tokenizer()` – gerenciam embeddings customizados.
- `modules/modelSetup/mixin/ModelSetupDebugMixin.py`
  - `_save_image()`, `_save_text()` e `_project_latent_to_image_sdxl()` – utilidades de depuração.
- `modules/dataLoader/StableDiffusionXLBaseDataLoader.py`
  - `create_dataset()` – compõe módulos do MGDS para preparar lotes.
  - `_preparation_modules()`, `_cache_modules()`, `_debug_modules()` – estágios do pipeline.
- `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`
  - `_enumerate_input_modules()`, `_load_input_modules()` – leitura de imagens e prompts.
  - `_mask_augmentation_modules()` e `_augmentation_modules()` – augmentações.
  - `_output_modules_from_out_names()` – ordena e produz dados finais.
- `modules/modelSampler/StableDiffusionXLSampler.py`
  - `__sample_base()` / `__sample_inpainting()` – rotinas de denoising.
  - `sample()` – expõe interface única de amostragem.
- `modules/modelSaver/stableDiffusionXL/StableDiffusionXLLoRASaver.py`
  - `__get_state_dict()` – agrega pesos LoRA e embeddings.
  - `__save_ckpt()`, `__save_safetensors()`, `__save_internal()` – diferentes formatos de saída.
  - `save()` – ponto de entrada para salvar o modelo treinado.

