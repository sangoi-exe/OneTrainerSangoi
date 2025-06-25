import torch
from collections import defaultdict
import time
from modules.util import torch_util  # Importamos nosso utilitário com o accelerator global

class TokenGradientAnalyzer:
    def __init__(self, tokenizer_g, log_interval=2, top_k=5):
        """
        Inicializa o analisador.
        :param tokenizer_g: O tokenizer do CLIP-G (geralmente tokenizer_2).
        :param log_interval: Com que frequência (em passos) logar as estatísticas.
        :param top_k: Quantos tokens de maior/menor gradiente mostrar.
        """
        self.accelerator = torch_util.accelerator  # Usa o singleton global
        self.tokenizer_g = tokenizer_g
        self.log_interval = log_interval
        self.top_k = top_k

        # Estruturas para agregar estatísticas
        self.impact_sums = defaultdict(float)
        self.grad_counts = defaultdict(int)
        self.last_log_time = time.time()
        # Atributos para o padrão "Armar e Detonar"
        self.armed = False
        self.pending_data = {}

    def is_armed(self):
        return self.armed

    def set_pending_analysis(self, step, batch, loss_uncond):
        """ARMA o analisador com os dados pré-gradiente."""
        self.pending_data = {"step": step, "batch": batch, "loss_uncond": loss_uncond}
        self.armed = True

    def analyze_and_log(self, model):
        """
        DETONA o analisador. Executa a análise de gradientes e loga, se necessário.
        Deve ser chamada DEPOIS do loss.backward().
        """
        # Ponto de verificação 1: A função foi chamada? E estava armada?
        if not self.is_armed():
            # Esta mensagem não deve aparecer se tudo estiver correto.
            # Se aparecer, o ciclo "armar" está falhando em algum lugar.
            print(f"DEBUG: analyze_and_log chamada mas NÃO ESTAVA ARMADO.", flush=True)
            return

        step = self.pending_data['step']

        # Ponto de verificação 2: Estamos em um passo de log?
        is_log_step = (step + 1) % self.log_interval == 0
        
        # Imprime o status do teste a cada passo para diagnóstico
        print(f"DEBUG: analyze_and_log [Step: {step}] -> is_log_step = {is_log_step} (log_interval = {self.log_interval})", flush=True)

        if is_log_step:
            print(f"--- [Token Impact Analysis] INICIANDO ANÁLISE COMPLETA PARA O STEP {step} ---", flush=True)
            with torch.no_grad():
                unwrapped_model = self.accelerator.unwrap_model(model)
                text_embeddings_g = unwrapped_model.text_encoder_2.get_input_embeddings()

                # Ponto de verificação 3: Os gradientes existem?
                if text_embeddings_g.weight.grad is None:
                    print(f"AVISO: Gradientes de texto são NULOS no step {step}. Pulando análise.", flush=True)
                    self.armed = False
                    self.pending_data = {}
                    return

                # Pega a loss não condicionada que foi "armada"
                loss_uncond = self.pending_data.get('loss_uncond')
                if loss_uncond is None:
                    print(f"AVISO: loss_uncond não foi encontrada para o step {step}. Usando 1.0 como fallback.", flush=True)
                    loss_uncond = torch.tensor(1.0, device=self.accelerator.device)
                else:
                    loss_uncond = loss_uncond.detach() + 1e-6

                grad_norms = torch.norm(text_embeddings_g.weight.grad.detach(), p=2, dim=1)
                
                input_ids_g = self.pending_data['batch']['tokens_2']

                for i in range(input_ids_g.shape[0]):
                    for token_id in input_ids_g[i]:
                        token_id_item = token_id.item()
                        guidance_impact = grad_norms[token_id_item].item() / loss_uncond.item()
                        self.impact_sums[token_id_item] += guidance_impact
                        self.grad_counts[token_id_item] += 1

                avg_impacts = {
                    tid: self.impact_sums[tid] / self.grad_counts[tid]
                    for tid in self.impact_sums if self.grad_counts[tid] > 0
                }

                for tok in [self.tokenizer_g.pad_token_id, self.tokenizer_g.bos_token_id, self.tokenizer_g.eos_token_id]:
                    if tok in avg_impacts:
                        del avg_impacts[tok]

                sorted_impacts = sorted(avg_impacts.items(), key=lambda item: item[1], reverse=True)

                steps_per_second = self.log_interval / (time.time() - self.last_log_time)
                self.last_log_time = time.time()

                log_str = f"\n--- [Token Impact Analysis @ Step {step+1} | {steps_per_second:.2f} steps/s | Uncond Loss: {loss_uncond.item():.4f}] ---\n"
                log_str += "  Top Impact (High Semantic Conflict / Key Drivers):\n"
                for token_id, avg_impact in sorted_impacts[:self.top_k]:
                    token_str = self.tokenizer_g.decode([token_id])
                    log_str += f"    - '{token_str}' (ID: {token_id}): {avg_impact:.6f}\n"

                log_str += "  Bottom Impact (Low Conflict / Aligned Tokens):\n"
                for token_id, avg_impact in sorted_impacts[-self.top_k:]:
                    token_str = self.tokenizer_g.decode([token_id])
                    log_str += f"    - '{token_str}' (ID: {token_id}): {avg_impact:.6f}\n"

                log_str += "---------------------------------------------------------"
                print(log_str, flush=True)
                print(f"--- [Token Impact Analysis] FIM DA ANÁLISE ---", flush=True)

        # Desarma para o próximo passo, independentemente de ter logado ou não.
        self.armed = False
        self.pending_data = {}