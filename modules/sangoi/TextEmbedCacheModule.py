from mgds.pipelineModule import PipelineModule
import torch
import pathlib

class TextEmbedCacheModule(PipelineModule):
    def __init__(self, model, config, cache_dir: str):
        super().__init__()
        self.model = model
        self.config = config
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Movemos os TEs para a GPU uma vez na inicialização
        self.model.text_encoder_1.to(self.model.unet.device)
        self.model.text_encoder_2.to(self.model.unet.device)

    def run(self, data: dict):
        # A lógica de cache só é executada se a flag `only_cache` estiver ativa.
        if not self.config.only_cache:
            return data

        image_path = data['image_path']

        # Verifica se o cache já existe para não reprocessar
        image_filename = pathlib.Path(image_path).stem
        cache_path = self.cache_dir / f"{image_filename}.pt"
        if cache_path.exists():
            return data # Já foi cacheado, apenas passa os dados adiante

        print(f"Gerando cache de texto para: {image_path}")

        # Pega os tokens diretamente do dicionário `data`
        tokens_1 = data['tokens_1']
        tokens_2 = data['tokens_2']

        # Garante que os tensores tenham uma dimensão de batch para o encode_text
        if tokens_1.ndim == 1:
            tokens_1 = tokens_1.unsqueeze(0)
        if tokens_2.ndim == 1:
            tokens_2 = tokens_2.unsqueeze(0)

        with torch.no_grad():
            text_encoder_output, pooled_output = self.model.encode_text(
                train_device=self.model.unet.device,
                batch_size=1,
                tokens_1=tokens_1,
                tokens_2=tokens_2
            )
        
        cache_data = {
            'encoder_hidden_states': text_encoder_output.cpu(),
            'pooled_output': pooled_output.cpu(),
            'input_ids_g': tokens_2.cpu(),
            'input_ids_l': tokens_1.cpu()
        }
        torch.save(cache_data, cache_path)

        # Retorna os dados originais para o próximo módulo da pipeline
        return data