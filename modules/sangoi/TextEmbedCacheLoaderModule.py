import torch
import pathlib
from mgds.pipelineModule import PipelineModule

class TextEmbedCacheLoaderModule(PipelineModule):
    def __init__(self, cache_dir: str):
        super().__init__()
        self.cache_dir = pathlib.Path(cache_dir)

    def run(self, data: dict):
        image_path = data['image_path']
        image_filename = pathlib.Path(image_path).stem
        cache_path = self.cache_dir / f"{image_filename}.pt"

        if cache_path.exists():
            cached_text_data = torch.load(cache_path)
            # Adiciona os dados cacheados ao batch
            data.update(cached_text_data)
            # Remove a legenda crua para garantir que não seja re-tokenizada
            data.pop('prompt', None)
        else:
            # Fallback caso o cache não exista para este item
            print(f"[AVISO] Cache de texto não encontrado para {image_filename}. O treino pode falhar.")
            
        return data