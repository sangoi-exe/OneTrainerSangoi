import torch
from typing import List, Optional, Dict, Any

class ProbeScheduler:
    """
    Agenda e executa "passos sonda" para medir o impacto de gradiente de tokens
    que não aparecem naturalmente no dataset de treino.
    """

    def __init__(
        self,
        tokenizer_l,  # Tokenizer para CLIP-L
        tokenizer_g,  # Tokenizer para CLIP-G
        token_analyzer, # Instância do TokenGradientAnalyzer
        probe_interval: int = 10, # A cada quantos passos reais fazer a sondagem
        probe_batch_size: int = 8, # Quantos tokens sondar por vez
    ):
        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g
        self.token_analyzer = token_analyzer
        self.probe_interval = probe_interval
        self.probe_batch_size = probe_batch_size

        # Fila de tokens a serem sondados
        self.probe_queue: List[int] = []
        self.probed_tokens: set[int] = set()

        self._initialize_probe_queue()

    def _initialize_probe_queue(self):
        """Preenche a fila com todos os tokens dos vocabulários."""
        vocab_l = set(self.tokenizer_l.get_vocab().keys()) if self.tokenizer_l else set()
        vocab_g = set(self.tokenizer_g.get_vocab().keys()) if self.tokenizer_g else set()
        
        # Usa os IDs, não os textos
        all_token_ids = set(self.tokenizer_l.get_vocab().values()) | set(self.tokenizer_g.get_vocab().values())

        # Remove tokens especiais para não sondá-los
        special_ids = set()
        for tokenizer in [self.tokenizer_l, self.tokenizer_g]:
            if tokenizer:
                special_ids.update([tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id])

        self.probe_queue = sorted(list(all_token_ids - special_ids))
        print(f"[ProbeScheduler] Fila de sondagem inicializada com {len(self.probe_queue)} tokens.")

    def should_probe(self, step: int) -> bool:
        """Verifica se é hora de rodar a sondagem."""
        return (step + 1) % self.probe_interval == 0 and self.probe_queue

    def run_probe(
        self, 
        model: torch.nn.Module, 
        device: torch.device,
        step: int
    ) -> None:
        """
        Executa um passo de sondagem sintético.
        """
        if not self.probe_queue:
            print("[ProbeScheduler] Fila de sondagem vazia. Nenhuma ação a tomar.")
            return

        # Pega o próximo lote de tokens da fila
        tokens_to_probe = [self.probe_queue.pop(0) for _ in range(min(self.probe_batch_size, len(self.probe_queue)))]
        
        print(f"[ProbeScheduler] Sondando {len(tokens_to_probe)} tokens no passo {step+1}...")

        # Cria um batch "dummy"
        # O prompt é simplesmente o token a ser sondado
        prompts = [self.tokenizer_g.decode(token_id) for token_id in tokens_to_probe]
        
        # Tokeniza para ambos encoders
        tokens_1 = self.tokenizer_l(prompts, padding="max_length", max_length=self.tokenizer_l.model_max_length, truncation=True, return_tensors="pt").input_ids
        tokens_2 = self.tokenizer_g(prompts, padding="max_length", max_length=self.tokenizer_g.model_max_length, truncation=True, return_tensors="pt").input_ids

        batch_size = len(tokens_to_probe)
        dummy_latents = torch.zeros((batch_size, 4, 128, 128), device=device) # Ajustar tamanho se necessário

        dummy_batch = {
            "tokens_1": tokens_1.to(device),
            "tokens_2": tokens_2.to(device),
            "latent_image": dummy_latents,
            "image_path": [f"probe_{tok}" for tok in tokens_to_probe],
            # Adicionar outras chaves que o modelo espera, com valores dummy
            "original_resolution": torch.tensor([[1024, 1024]] * batch_size, device=device),
            "crop_offset": torch.tensor([[0, 0]] * batch_size, device=device),
            "crop_resolution": torch.tensor([[1024, 1024]] * batch_size, device=device),
        }

        # Garante que o modelo está em modo de avaliação para não afetar batchnorm/dropout, etc.
        original_mode = model.text_encoder_2.training
        model.text_encoder_1.eval()
        model.text_encoder_2.eval()

        # Roda forward e backward sem otimizador para obter gradientes
        with torch.set_grad_enabled(True):
            model.zero_grad()
            # A perda aqui é arbitrária, apenas para gerar gradientes. Usamos 1.0.
            dummy_loss = model(dummy_batch).mean() # Simula uma perda
            dummy_loss.backward()

            # Arma e dispara o analisador de token
            self.token_analyzer.set_pending_analysis(step, dummy_batch, loss_uncond=torch.tensor(1.0))
            self.token_analyzer.analyze_and_log(model)

            # Limpa os gradientes para não interferir no passo de treino real
            model.zero_grad()

        # Restaura o modo original do modelo
        model.text_encoder_1.train(original_mode)
        model.text_encoder_2.train(original_mode)

        # Adiciona os tokens sondados à lista de já vistos
        self.probed_tokens.update(tokens_to_probe)
        print(f"[ProbeScheduler] Sondagem concluída. {len(self.probe_queue)} tokens restantes na fila.")
