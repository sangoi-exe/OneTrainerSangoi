import re
import os
import torch
import pathlib
import numpy as np
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.util.time_util import get_string_timestamp
from typing import Dict, Any, List, Optional, Tuple

# Assumindo que estas classes estão em um módulo acessível, como você definiu.
from modules.modelSetup.BaseStableDiffusionXLSetup import AttentionMapLogger, CaptureManager
from diffusers.models.attention_processor import Attention

class CrossAttnMapsAnalyzer:
    """
    Calcula afinidade de tokens via gradiente e/ou atenção.
    Usa um sistema de captura de atenção robusto e desacoplado.
    """

    def __init__(
        self,
        config,
        model,
        tokenizer_l,
        tokenizer_g,
        out_dir: str = "token_affinity_reports",
    ):
        """
        Construtor simplificado e focado no essencial.
        """
        # --- Configurações Essenciais ---
        self.enable_attn_report = config.analyzer_enable_attn_report
        self.enable_grad_report = config.analyzer_enable_grad_report
        self.heatmap_interval = config.analyzer_heatmap_interval
        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g

        # --- Ferramentas e Estado ---
        self.map_logger = AttentionMapLogger()
        self.cap_manager = CaptureManager(self.map_logger)
        
        self._grad_l: Optional[torch.Tensor] = None
        self._grad_g: Optional[torch.Tensor] = None
        self.pending_data: Dict[str, Any] = {}
        
        # --- Setup do Diretório de Saída ---
        base_out_dir = pathlib.Path(out_dir)
        timestamp = get_string_timestamp(style='file')
        self.out_dir = base_out_dir / timestamp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TokenAnalyzer] Relatórios serão salvos em: {self.out_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TokenAnalyzer] Inicializado. Device: {self.device}")

        self.prompt_template: Dict[str, Any] = {
            "token_ids": None,
            "token_texts": None
        }
        self.template_set = False

        self.layer_names = [
            name.replace('.', '_')
            for name, m in model.unet.named_modules()
            if isinstance(m, Attention) and "attn2" in name
        ]

    def set_pending_analysis(self, step: int, batch: Dict[str, Any]):
        """Prepara para um novo passo de análise. Limpa o estado antigo."""
        self._grad_l, self._grad_g = None, None
        # O logger é limpo pelo AttnCaptureManager, mas uma limpeza extra aqui não faz mal.
        self.map_logger.clear()
        self.pending_data = {"step": step, "batch": batch}

    def analyze_gradients_after_backward(self, model: torch.nn.Module):
        """Chamado após loss.backward() para analisar gradientes."""
        if not self.enable_grad_report:
            return

        step = self.pending_data.get("step", -1)
        batch = self.pending_data.get("batch", {})
        if not batch: return

        try:
            self._analyze_tokens_from_gradient(model, step, batch)
        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao analisar gradientes (Step {step}): {e}")
            import traceback
            traceback.print_exc()

    def analyze_attention_after_step(self, attn_maps_to_analyze, current_epoch: int):
        """Chamado no final do passo para analisar a atenção."""
        step = self.pending_data.get("step", -1)
        batch = self.pending_data.get("batch", {})
        if not batch: return

        try:
            # Análise de Atenção para scores
            if self.enable_attn_report:
                self._analyze_attention_scores(attn_maps_to_analyze, step, batch)
            
            # Geração de Heatmaps em intervalos
            is_heatmap_step = self.heatmap_interval > 0 and (current_epoch % self.heatmap_interval == 0)
            if is_heatmap_step:
                self._visualize_attention_heatmaps(attn_maps_to_analyze, step, batch)
        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao analisar atenção/heatmaps (Step {step}): {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.pending_data.clear()

    def _analyze_tokens_from_gradient(self, model, step: int, batch: Dict[str, Any]):
        """
        Função que analisa a atribuição de gradiente.
        """
        grad_l, grad_g = None, None

        if self._grad_l is not None and self._grad_g is not None:
            grad_l, grad_g = -self._grad_l, -self._grad_g
        else:
            cached_g = batch.get('text_encoder_2_hidden_state')
            if cached_g is not None and cached_g.grad is not None:
                grad_g = -cached_g.grad
            
            cached_l = batch.get('text_encoder_1_hidden_state')
            if cached_l is not None and cached_l.grad is not None:
                grad_l = -cached_l.grad

        if grad_g is None:
            print(f"[TokenAnalyzer] Fonte de gradiente não encontrada no step {step}. Pulando análise de gradiente.")
            return

        scores_g = grad_g.norm(p=2, dim=-1)
        grad_scores = scores_g.mean(dim=0)
        if grad_l is not None:
            scores_l = grad_l.norm(p=2, dim=-1)
            grad_scores = (grad_scores + scores_l.mean(dim=0)) / 2

        self._save_prompt_based_report(step, batch, grad_scores, "Gradient_Attribution")

    def _analyze_attention_scores(self, maps, step: int, batch: Dict[str, Any]):
        """
        Calcula um score de atenção agregado por token, lidando com as diferentes
        resoluções espaciais das camadas da UNet.
        """
        attn_maps_raw = maps
        if not attn_maps_raw:
            print(f"[TokenAnalyzer] SEM ATTENTION MAP PRO ANALYZE SCORES")
            return

        base_batch_size = batch['tokens_1'].shape[0]
        layer_scores = []

        for raw_map in attn_maps_raw:
            # raw_map tem shape (B_eff, H, Q, K)
            # B_eff é o batch que a UNet viu (provavelmente 2 com CFG)
            cond_map = raw_map[:base_batch_size] # Pega a fatia condicional
            score_per_token = cond_map.sum(dim=(0, 1, 2)) # Soma em B, H, Q -> Shape (K,)
            layer_scores.append(score_per_token)

        if not layer_scores: return

        total_scores = torch.stack(layer_scores).sum(dim=0)
        
        if total_scores.ndim == 0:
            print("[TokenAnalyzer] Scores de atenção agregados resultaram em um escalar. Pulando.")
            return

        self._save_prompt_based_report(step, batch, total_scores, "Attention_Attribution")

    def _visualize_attention_heatmaps(self, attn_maps_raw, step: int, batch: Dict[str, Any]):
        """
        Orquestrador principal: agrega os mapas e chama a função de plotagem
        para ambos os conjuntos de tokens (CLIP-L e CLIP-G).
        """
        if not attn_maps_raw:
            print(f"[TokenAnalyzer] SEM ATTENTION MAP PRO HEATMAP")
            return

        try:
            base_batch_size = batch['tokens_1'].shape[0]
            conditional_maps = [m[:base_batch_size] for m in attn_maps_raw]

            latent_h, latent_w = batch['latent_image'].shape[2], batch['latent_image'].shape[3]
            aspect_ratio = latent_h / latent_w if latent_w > 0 else 1.0

            min_q_dim = min(m.shape[2] for m in conditional_maps)
            target_h, target_w = self._infer_spatial_dims(min_q_dim, aspect_ratio)

            if target_h == -1: return

            normalized_maps = []
            for cond_map in conditional_maps:
                map_avg_heads = cond_map.mean(dim=1)
                current_q_dim = map_avg_heads.shape[1]
                current_h, current_w = self._infer_spatial_dims(current_q_dim, aspect_ratio)
                if current_h == -1: continue
                
                num_tokens = map_avg_heads.shape[2]
                map_reshaped = map_avg_heads.permute(0, 2, 1).view(1, num_tokens, current_h, current_w)
                map_resized = F.interpolate(map_reshaped, size=(target_h, target_w), mode='bilinear', align_corners=False)
                normalized_maps.append(map_resized)

            if not normalized_maps: return

            aggregated_heatmap_data = torch.stack(normalized_maps).mean(dim=0).squeeze(0).to(torch.float32).cpu().numpy()

            # --- ETAPA 2: LÓGICA DE GABARITO E ORDENAÇÃO ---
            concept_name = batch.get("concept", "default")[0]
            current_token_ids = batch['tokens_2'][0] # Usa CLIP-G

            # Se este é o primeiro conceito 'orig', salvamos como gabarito.
            if concept_name == 'orig' and not self.template_set:
                self.prompt_template["token_ids"] = current_token_ids.cpu().numpy()
                self.prompt_template["token_texts"] = [self.tokenizer_g.decode(tid) for tid in current_token_ids]
                self.template_set = True
                print(f"[ANALYSIS] Gabarito de prompt 'orig' salvo no step {step}.")

            # --- ETAPA DE ORQUESTRAÇÃO DA PLOTAGEM ---
            _, image_tag = self._get_common_report_header(step, batch, "")

            if self.template_set and concept_name != 'orig':
                # Mapa de reordenação: {id_do_token_gabarito: indice_do_heatmap_atual}
                current_id_to_idx = {tid.item(): i for i, tid in enumerate(current_token_ids)}
                
                # Cria um novo tensor de dados de heatmap na ordem do gabarito.
                ordered_heatmap_data = np.zeros_like(aggregated_heatmap_data)
                for i, template_id in enumerate(self.prompt_template["token_ids"]):
                    if template_id in current_id_to_idx:
                        current_idx = current_id_to_idx[template_id]
                        ordered_heatmap_data[i] = aggregated_heatmap_data[current_idx]
                
                # Plota usando os dados reordenados e os tokens do gabarito.
                self._plot_and_save_heatmap(
                    step, image_tag, ordered_heatmap_data, 
                    self.prompt_template["token_ids"], self.prompt_template["token_texts"], 
                    f"{concept_name}_vs_orig"
                )
            else:
                # Se for o 'orig' ou se não houver gabarito, plota na ordem normal.
                token_texts = [self.tokenizer_g.decode(tid) for tid in current_token_ids]
                self._plot_and_save_heatmap(
                    step, image_tag, aggregated_heatmap_data, 
                    current_token_ids.cpu().numpy(), token_texts, 
                    concept_name
                )

        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao gerar heatmap de atenção: {e}")
            import traceback
            traceback.print_exc()

    def _plot_and_save_heatmap(self, step: int, image_tag: str, heatmap_data: np.ndarray, 
                              token_ids: np.ndarray, token_texts: list, output_suffix: str):
        """
        Plota um grid de heatmaps com layout adaptativo para diferentes aspect ratios.
        """
        # 1. Filtra os tokens que realmente importam (sua lógica atual está boa)
        tokens_to_plot = []
        indices_to_plot = []
        for i, token_text in enumerate(token_texts):
            token_id = token_ids[i]
            is_special = token_id in [self.tokenizer_g.eos_token_id, self.tokenizer_g.pad_token_id, self.tokenizer_g.bos_token_id]
            if is_special: continue
            if token_text.strip() == '': continue
            tokens_to_plot.append(token_text.replace('</w>', '').strip())
            indices_to_plot.append(i)

        if not tokens_to_plot: return

        heatmaps_to_plot = heatmap_data[indices_to_plot, :, :]
        num_tokens_to_plot = len(tokens_to_plot)

        # 2. LÓGICA DE LAYOUT ADAPTATIVO
        # Pega o aspect ratio do *heatmap em si* (H/W)
        heatmap_h, heatmap_w = heatmaps_to_plot[0].shape
        heatmap_aspect_ratio = heatmap_h / heatmap_w if heatmap_w > 0 else 1.0

        # Define as colunas (você ainda controla isso)
        cols = 12
        # Calcula as linhas necessárias
        rows = (num_tokens_to_plot + cols - 1) // cols

        # Ajusta o `figsize` com base no aspect ratio dos subplots e do grid
        # A ideia é dar mais espaço vertical para imagens em modo retrato e vice-versa.
        # A largura da figura é proporcional ao número de colunas.
        # A altura da figura é proporcional ao número de linhas E ao aspect ratio dos heatmaps.
        fig_width = cols * 2.5
        fig_height = rows * (2.5 * heatmap_aspect_ratio) + 1.5 # +1.5 para títulos e margens
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=120)
        fig.suptitle(f'Attention Heatmap ({output_suffix}) - Step {step}', fontsize=16)
        
        axes_flat = axes.flat if num_tokens_to_plot > 1 else [axes]

        # 3. Plota cada heatmap
        for i in range(num_tokens_to_plot):
            ax = axes_flat[i]
            heatmap = heatmaps_to_plot[i, :, :]
            ax.imshow(heatmap, cmap='cividis')
            # Adiciona o índice original para referência
            ax.set_title(f'{indices_to_plot[i]}: "{tokens_to_plot[i]}"', fontsize=8)
            ax.axis('off')

        # 4. Esconde eixos de subplots não utilizados
        for i in range(num_tokens_to_plot, len(axes_flat)):
            axes_flat[i].axis('off')

        # 5. Usa `plt.subplots_adjust` para um controle mais fino que `tight_layout`
        # Isso ajuda a evitar a sobreposição de títulos em imagens quadradas.
        plt.subplots_adjust(
            left=0.02, 
            right=0.98, 
            top=0.92 if fig_height > 5 else 0.85, # Deixa mais espaço para o título principal em figuras altas
            bottom=0.02,
            hspace=0.4, # Aumenta o espaço horizontal entre os plots
            wspace=0.1  # Aumenta o espaço vertical entre os plots
        )
        
        # 6. Salva como JPEG
        outfile = self.out_dir / f"{image_tag}_HEATMAP_step_{step:06d}_{output_suffix}.jpg"
        plt.savefig(outfile, format='jpeg', dpi=96, pil_kwargs={'quality': 85})
        plt.close(fig)

    @staticmethod
    def _infer_spatial_dims(q_dim: int, aspect_ratio: float) -> Tuple[int, int]:
        """
        Encontra os fatores inteiros H e W de q_dim que melhor se aproximam
        do aspect_ratio fornecido.
        Isto é determinístico e não usa aproximações de float.
        """
        best_h, best_w = -1, -1
        min_ratio_diff = float('inf')

        # Itera de sqrt(q_dim) para baixo, que é a forma mais eficiente de encontrar fatores.
        for w in range(int(np.sqrt(q_dim)), 0, -1):
            if q_dim % w == 0:
                h = q_dim // w
                # Agora temos um par de fatores (h, w).
                # Precisamos verificar qual orientação (h/w ou w/h) está mais próxima do aspect_ratio.
                
                # Checa a orientação 1 (h/w)
                ratio1 = h / w
                diff1 = abs(ratio1 - aspect_ratio)
                if diff1 < min_ratio_diff:
                    min_ratio_diff = diff1
                    best_h, best_w = h, w

                # Checa a orientação 2 (w/h)
                ratio2 = w / h
                diff2 = abs(ratio2 - aspect_ratio)
                if diff2 < min_ratio_diff:
                    min_ratio_diff = diff2
                    best_h, best_w = w, h
        
        return best_h, best_w

    def _save_prompt_based_report(self, step: int, batch: Dict[str, Any], scores: torch.Tensor, report_type: str):
        """
        Salva um relatório simples mostrando a importância dos tokens no prompt atual.
        """
        report_header, image_tag = self._get_common_report_header(step, batch, report_type)
        
        token_ids = batch['tokens_2'][0] # Usa CLIP-G como referência
        tokens_text = [self.tokenizer_g.decode(tid) for tid in token_ids]

        scored_tokens = []
        for i, score in enumerate(scores):
            token_id = token_ids[i].item()
            token_text = tokens_text[i].replace('</w>', '').strip()
            
            if token_text in [self.tokenizer_g.eos_token, self.tokenizer_g.pad_token, self.tokenizer_g.bos_token, '']:
                continue
            
            scored_tokens.append({'text': token_text, 'id': token_id, 'score': score.item()})

        sorted_tokens = sorted(scored_tokens, key=lambda x: x['score'], reverse=True)

        report_lines = list(report_header)
        report_lines.append(f"--- Importância de Tokens no Prompt ({report_type}) ---\n")
        report_lines.append("Score      | Token (ID)\n")
        report_lines.append("-----------|----------------------------------\n")
        
        for item in sorted_tokens:
            report_lines.append(f"{item['score']:<10.6f} | {item['text']} (ID: {item['id']})\n")

        safe_report_type = report_type.replace(" ", "_")
        outfile = self.out_dir / f"{image_tag}_step_{step:06d}_{safe_report_type}.txt"
        outfile.write_text("".join(report_lines), encoding="utf-8", errors="ignore")

    def _get_common_report_header(self, step: int, batch: Dict[str, Any], report_type: str) -> Tuple[List[str], str]:
        """Gera o cabeçalho padrão para os arquivos de relatório."""
        image_paths_val = batch.get("image_path", ["N/A"])
        image_tag = self._sanitize_filename(image_paths_val[0]) if image_paths_val else f"batch_step_{step}"
        
        report_header = [
            f"### Relatório de {report_type} - Step: {step}\n",
            f"### Imagem: {image_paths_val[0]}\n",
            "-------------------------------------------------\n\n"
        ]
        return report_header, image_tag

    @staticmethod
    def _sanitize_filename(candidate: str) -> str:
        """Limpa um nome de arquivo para ser seguro para o sistema de arquivos."""
        basename = os.path.splitext(os.path.basename(candidate))[0]
        return re.sub(r"[^\w.\-]+", "_", basename)

    # --- Métodos de Ciclo de Vida ---
    def prepare_for_step(self, step: int, batch: dict):
        self.pending_data = {"step": step, "batch": batch}
        self.map_logger.clear() # Limpa o logger para o novo passo de captura
        
    def calculate_attention_entropy(self, batch, attn_maps_raw: list) -> float | None:
        """
        Calcula a entropia média das camadas de atenção capturadas.
        Esta função é "read-only" e retorna um valor float ou None.
        """
        if not attn_maps_raw:
            print("[Entropy] Nenhum mapa de atenção para calcular a entropia.")
            return None

        # O batch size precisa ser obtido dos dados pendentes, que ainda devem estar disponíveis        
        if not batch:
            print("[Entropy] Batch é None.")
            return None
        
        base_batch_size = batch.get('tokens_1', torch.empty(0)).shape[0]
        if base_batch_size == 0: return None

        try:
            entropies_per_layer = []
            
            # O cálculo é feito fora do grafo
            with torch.no_grad():
                for raw_map in attn_maps_raw:
                    p_cond = raw_map[:base_batch_size].float()
                    p_cond_clamped = torch.clamp(p_cond, min=1e-9)
                    entropy_map = -(p_cond_clamped * torch.log2(p_cond_clamped)).sum(dim=-1)
                    avg_entropy_for_layer = entropy_map.mean()
                    entropies_per_layer.append(avg_entropy_for_layer)

            if not entropies_per_layer:
                return None

            # Retorna o valor escalar final
            return torch.stack(entropies_per_layer).mean().item()

        except Exception as e:
            print(f"[Entropy] ERRO ao calcular a entropia da atenção: {e}")
            return None
        
    def get_per_head_entropy(self, attn_maps_raw, batch):
        if not attn_maps_raw: return None
        B = batch.get('tokens_1', torch.empty(0)).size(0)
        if B == 0: return None

        per_head = {}
        for layer_name, raw in zip(self.layer_names, attn_maps_raw):
            p = raw[:B].float().clamp_min_(1e-9)
            ent = -(p * p.log2()).sum(-1) # (B, H, Q)
            ent_h = ent.mean(dim=(0, 2)) # (H,)
            for h, v in enumerate(ent_h):
                per_head[f"{layer_name}_h{h:02d}"] = v.item()
        return per_head