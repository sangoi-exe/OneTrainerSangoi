import re
import os
import torch
import pathlib
import numpy as np
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from modules.util.time_util import get_string_timestamp
from typing import Dict, Any, List, Optional, Tuple, Union

from diffusers.models.attention_processor import Attention
from modules.modelSetup.BaseStableDiffusionXLSetup import AttentionMapLogger, CapturingAttnProcessor

class TokenGradientAnalyzer:
    """
    Calcula afinidade de tokens via gradiente E atenção.
    A análise de atenção agora é robusta e desacoplada, usando um sistema de
    substituição de processador em vez de hooks frágeis.
    """

    def __init__(
        self,
        tokenizer_l,
        tokenizer_g,
        top_k_tokens: int = 5000,
        out_dir: str = "token_affinity_reports",
        *,
        tag_file: Optional[str] = None,
        tag_cache_file: str = "tag_embedding_cache.pt",
        top_n_tags: int = 100,
        ema_beta: float = 0.9,
        report_global_stats_top_n: int = 200,
    ):
        if not (0 < ema_beta < 1):
            raise ValueError("ema_beta deve estar entre 0 e 1")

        # --- Configurações Gerais (Mantidas) ---
        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g
        self.top_k_tokens = top_k_tokens
        self.out_dir = pathlib.Path(out_dir)
        self.report_global_stats_top_n = report_global_stats_top_n
        self.tag_file = pathlib.Path(tag_file) if tag_file else None
        self.tag_cache_file = pathlib.Path(tag_cache_file)
        self.top_n_tags = top_n_tags
        self._tags_text_list: List[str] = []
        self._tag_embeddings_matrix: Optional[torch.Tensor] = None

        # --- Estado da Análise de Gradiente ---
        self._grad_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._grad_l: Optional[torch.Tensor] = None
        self._grad_g: Optional[torch.Tensor] = None

        # --- Estado da Análise de Atenção (A FORMA CORRETA) ---
        self.map_logger = AttentionMapLogger()
        self._original_attn_processors: Dict[str, Any] = {}
        self.is_capturing_attention = False

        # --- Estado de Controle ---
        self.pending_data: Dict[str, Any] = {}

        # --- Métricas (SEPARADAS para clareza e correção) ---
        self.ema_beta = ema_beta
        # Scores de Gradiente
        self.grad_token_ema_scores: Dict[int, float] = {}
        self.grad_token_all_scores_history: Dict[int, List[float]] = defaultdict(list)
        # Scores de Atenção
        self.attn_token_ema_scores: Dict[int, float] = {}
        self.attn_token_all_scores_history: Dict[int, List[float]] = defaultdict(list)
        # Scores de Tags (baseado em gradiente)
        self.tag_ema_scores: Dict[str, float] = {}
        self.tag_all_scores_history: Dict[str, List[float]] = defaultdict(list)
        
        base_out_dir = pathlib.Path(out_dir)
        timestamp = get_string_timestamp(style='file') # Ex: '20250622_143055'
        self.out_dir = base_out_dir / timestamp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TokenAnalyzer] Relatórios serão salvos em: {self.out_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TokenAnalyzer] Inicializado. Device: {self.device}")

    def _backward_hook_l(self, module, grad_input, grad_output):
        if grad_output[0] is not None: self._grad_l = grad_output[0].detach()

    def _backward_hook_g(self, module, grad_input, grad_output):
        if grad_output[0] is not None: self._grad_g = grad_output[0].detach()

    def start_analysis_hooks(self, model: torch.nn.Module):
        """
        Prepara o modelo para uma rodada de análise.
        1. Registra hooks de gradiente.
        2. Substitui os processadores de atenção para iniciar a captura.
        """
        if model is None:
            print("[TokenAnalyzer] Modelo None. Análise não pode ser iniciada.")
            return

        # 1. Hooks de Gradiente
        self._grad_hook_handles.clear()
        try:
            self._grad_hook_handles.append(
                model.text_encoder_1.text_model.embeddings.register_full_backward_hook(self._backward_hook_l)
            )
            self._grad_hook_handles.append(
                model.text_encoder_2.text_model.embeddings.register_full_backward_hook(self._backward_hook_g)
            )
        except AttributeError as e:
            print(f"[TokenAnalyzer] Erro registrando hooks de gradiente: {e}")

        # 2. Captura de Atenção via substituição de processador
        if not self.is_capturing_attention:
            print("[TokenAnalyzer] Iniciando captura de atenção...")
            
            # Salva os processadores originais para restauração, como antes.
            self._original_attn_processors = model.unet.attn_processors.copy()
            
            capturing_processor = CapturingAttnProcessor(self.map_logger)
            
            attn_module_names_to_capture = [
                name for name, module in model.unet.named_modules() 
                if isinstance(module, Attention) and "attn2" in name
            ]

            if not attn_module_names_to_capture:
                print("[AVISO] Nenhum módulo de cross-attention ('attn2') encontrado para substituição.")
                return

            # AGORA VOCÊ ITERA SOBRE A LISTA CORRETA.
            for name in attn_module_names_to_capture:
                model.unet.set_processor(name, capturing_processor)
            
            self.is_capturing_attention = True

    def stop_analysis_hooks(self, model: torch.nn.Module):
        """Limpa tudo e restaura o modelo ao seu estado original."""
        # 1. Remove hooks de gradiente
        for h in self._grad_hook_handles:
            h.remove()
        self._grad_hook_handles.clear()

        # 2. Restaura processadores de atenção
        if self.is_capturing_attention:
            print("[TokenAnalyzer] Restaurando processadores de atenção originais...")
            if self._original_attn_processors:
                model.unet.set_attn_processor(self._original_attn_processors)
            self._original_attn_processors.clear()
            self.map_logger.clear()
            self.is_capturing_attention = False
            print("[TokenAnalyzer] Processadores restaurados.")

    def set_pending_analysis(self, step: int, batch: Dict[str, Any]):
        self._grad_l, self._grad_g = None, None
        self.map_logger.clear() # Limpa mapas da iteração anterior
        self.pending_data = {"step": step, "batch": batch}

    def analyze_and_save_report(self, model: torch.nn.Module):
        step = self.pending_data.get("step", -1)
        batch = self.pending_data.get("batch", {})

        try:
            # Análise de Gradiente (sua lógica original, intocada)
            if self._grad_l is not None and self._grad_g is not None:
                grad_l_full, grad_g_full = -self._grad_l, -self._grad_g
                with torch.no_grad():
                    self._analyze_tokens_from_gradient(model, grad_l_full, grad_g_full, step, batch)
            else:
                print(f"[TokenAnalyzer] Gradientes não capturados no step {step}.")

            # Análise de Atenção (agora consome do logger)
            self._analyze_tokens_from_attention(step, batch)
            self._visualize_attention_heatmaps(step, batch)

        except Exception as e:
            import traceback
            print(f"[TokenAnalyzer] ERRO inesperado (step {step}): {e}")
            traceback.print_exc()
        finally:
            self._grad_l, self._grad_g = None, None
            self.pending_data.clear()

    def _analyze_tokens_from_gradient(self, model, grad_l, grad_g, step, batch):
        device = grad_g.device
        emb_l = model.text_encoder_1.get_input_embeddings().weight.to(device)
        emb_g = model.text_encoder_2.get_input_embeddings().weight.to(device)

        vec_l = grad_l.mean(dim=(0, 1))
        vec_g = grad_g.mean(dim=(0, 1))

        affinity_l = torch.matmul(emb_l, vec_l)
        affinity_g = torch.matmul(emb_g, vec_g)

        grad_scores = affinity_g if affinity_l.shape != affinity_g.shape else affinity_l + affinity_g
        if affinity_l.shape != affinity_g.shape:
            print(f"[TokenAnalyzer] AVISO (Step {step}): vocab L≠G, usando só G.")

        self._update_all_score_metrics(grad_scores, is_token=True, use_attention_scores=False)
        self._save_report(step, batch, grad_scores, "Gradient Tokens", self.grad_token_ema_scores)

    def _analyze_tokens_from_attention(self, step: int, batch: Dict[str, Any]):
        """
        Calcula um score de atenção agregado por token, lidando com as diferentes
        resoluções espaciais das camadas da UNet.
        """
        # Pega os mapas crus (lista de tensores (B, H, Q, K))
        attn_maps_raw = self.map_logger.get_maps()
        if not attn_maps_raw:
            print(f"[TokenAnalyzer] SEM ATTENTION MAP PRO ANALYZE TOKENS")
            return

        try:
            # O batch size que você ACHA que tem (geralmente 1)
            base_batch_size = batch['tokens_1'].shape[0]

            # Lista para guardar os vetores de score de cada camada
            layer_scores = []

            for cond_map in attn_maps_raw:
                # cond_map tem shape (B, H, Q, K)
                # O B aqui pode ser 2 (do CFG). Nós só queremos a parte condicional.
                
                # 1. Extrai a fatia condicional do batch.
                # O resultado terá shape (base_batch_size, H, Q, K), ex: (1, H, Q, K)
                map_cond_only = cond_map[:base_batch_size]

                # 2. Calcula o score por token PARA ESTA CAMADA.
                # Soma a atenção em todas as outras dimensões (batch, heads, queries espaciais).
                # O resultado é um vetor de scores, um para cada token. Shape: (K,)
                score_per_token_for_layer = map_cond_only.sum(dim=(0, 1, 2))
                
                layer_scores.append(score_per_token_for_layer)

            if not layer_scores:
                print("[TokenAnalyzer] Nenhum score de camada pôde ser calculado.")
                return

            # 3. Agrega os scores de todas as camadas.
            # Agora todos os tensores em `layer_scores` têm o mesmo shape (K,), ex: (77,).
            # O stack vai funcionar perfeitamente.
            # stack -> (num_camadas, K) -> sum -> (K,)
            # Agora todos os tensores em layer_scores são vetores (K,). O stack funciona.
            total_scores = torch.stack(layer_scores).sum(dim=0)
            
            # VERIFICAÇÃO DE SANIDADE para evitar o erro de len()
            if total_scores.ndim == 0: # Se for um escalar
                print("[AVISO] Scores de atenção agregados resultaram em um escalar. Pulando análise.")
                return

            self._update_all_score_metrics(total_scores, is_token=True, use_attention_scores=True)
            self._save_report(step, batch, total_scores, "Attention_Scores", self.attn_token_ema_scores)

        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao analisar scores de atenção: {e}")
            import traceback
            traceback.print_exc()

    def _update_all_score_metrics(self, current_scores: torch.Tensor, is_token: bool, use_attention_scores: bool = False, item_list_for_keys: Optional[List[str]] = None):
        if is_token:
            ema_dict = self.attn_token_ema_scores if use_attention_scores else self.grad_token_ema_scores
            history_dict = self.attn_token_all_scores_history if use_attention_scores else self.grad_token_all_scores_history
        else: # Tags
            ema_dict = self.tag_ema_scores
            history_dict = self.tag_all_scores_history
        
        for i in range(len(current_scores)):
            score_val = current_scores[i].item()
            key = i if is_token else item_list_for_keys[i]
            
            # Sua "EMA" que é uma soma. Mantive, mas saiba que está errado.
            ema_dict[key] = ema_dict.get(key, 0.0) + score_val
            history_dict[key].append(score_val)

    # ========================================================================= #
    # MÉTODOS DE RELATÓRIO E UTILITÁRIOS (Refatorados para clareza)
    # ========================================================================= #

    def _save_report(self, step: int, batch: Dict[str, Any], current_scores: torch.Tensor, report_type: str, ema_scores: Dict):
        """Função unificada para salvar relatórios de step."""
        report_header, image_tag = self._get_common_report_header(step, batch, report_type)
        
        # O nome do arquivo agora inclui o tipo de score
        safe_report_type = report_type.replace(" ", "_")
        outfile = self.out_dir / f"{image_tag}_step_{step:06d}_{safe_report_type}.txt"

        # Prepara as linhas do relatório
        report_lines = list(report_header)
        report_lines.append(f"--- TOP {self.top_k_tokens} TOKENS (Score EMA Acumulado) ---\n")
        report_lines.extend(self._format_report_lines(ema_scores, is_token=True, top_n=self.top_k_tokens, score_type="EMA"))
        report_lines.append("\n")
        report_lines.append(f"--- TOP {self.top_k_tokens} TOKENS (Score Atual) ---\n")
        report_lines.extend(self._format_report_lines(current_scores, is_token=True, top_n=self.top_k_tokens, score_type="Atual"))

        try:
            outfile.write_text("".join(report_lines), encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[TokenAnalyzer] ERRO salvando relatório '{report_type}' (Step {step}): {e}")

    def _format_report_lines(self, scores: Union[torch.Tensor, Dict], is_token: bool, top_n: int, score_type: str) -> List[str]:
        if isinstance(scores, torch.Tensor):
            # Ordena por score atual
            sorted_indices = torch.argsort(scores, descending=True)
            items = [(idx.item(), scores[idx].item()) for idx in sorted_indices[:top_n]]
        else: # É um dicionário (EMA)
            # Ordena por score do dicionário
            sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            items = sorted_items[:top_n]

        if not items: return [f"Nenhuma pontuação {score_type} para reportar.\n"]
        
        header = f"Score {score_type:<10}| Token (ID)\n" if is_token else f"Score {score_type:<10}| Tag\n"
        lines = [header, "-----------|----------------------------------\n"]
        
        for key, score in items:
            text = self._decode_token_id(int(key)) if is_token else str(key)
            id_suffix = f" (ID: {key})" if is_token else ""
            lines.append(f"{score:<10.6f} | {text}{id_suffix}\n")
        return lines

    def _decode_token_id(self, token_id: int) -> str:
        # (Sua implementação original, sem mudanças)
        decoded_text_g, decoded_text_l = "", ""
        try:
            res_g = self.tokenizer_g.decode([token_id], skip_special_tokens=True).strip()
            if res_g and not res_g.startswith(("[UNK]", "<|", "</")): decoded_text_g = res_g
        except Exception: pass
        try:
            res_l = self.tokenizer_l.decode([token_id], skip_special_tokens=True).strip()
            if res_l: decoded_text_l = res_l
        except Exception: pass
        if decoded_text_g: return decoded_text_g
        if decoded_text_l: return decoded_text_l
        return f"[ID_NODECODE:{token_id}]"

    def _get_common_report_header(self, step: int, batch: Dict[str, Any], report_type: str) -> Tuple[List[str], str]:
        # (Sua implementação original, sem mudanças)
        image_paths_val = batch.get("image_path", batch.get("image_paths", ["N/A"]))
        if not isinstance(image_paths_val, list): image_paths_val = [str(image_paths_val)]
        image_tag = self._sanitize_filename(image_paths_val[0]) if image_paths_val and image_paths_val[0] not in ["N/A", ""] else f"batch_step_{step}"
        
        report_header = [
            f"### Relatório de {report_type} - Step: {step}\n",
            f"### Imagem(ns): {', '.join(image_paths_val)}\n",
            "-------------------------------------------------\n\n"
        ]
        return report_header, image_tag

    @staticmethod
    def _sanitize_filename(candidate: str) -> str:
        basename = os.path.splitext(os.path.basename(candidate))[0]
        return re.sub(r"[^\w.\-]+", "_", basename)


    def _visualize_attention_heatmaps(self, step: int, batch: Dict[str, Any]):
        """
        Orquestrador principal: agrega os mapas e chama a função de plotagem
        para ambos os conjuntos de tokens (CLIP-L e CLIP-G).
        """
        attn_maps_raw = self.map_logger.get_maps()
        if not attn_maps_raw:
            print(f"[TokenAnalyzer] SEM ATTENTION MAP PRO HEATMAP")
            return

        try:
            # --- ETAPA DE AGREGAÇÃO (como antes, mas agora é reutilizada) ---
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

            # --- ETAPA DE ORQUESTRAÇÃO DA PLOTAGEM ---
            _, image_tag = self._get_common_report_header(step, batch, "")

            # Plota para tokens_2 (CLIP-G), o mais importante
            self._plot_and_save_heatmap(step, image_tag, aggregated_heatmap_data, 
                                      batch['tokens_2'][0], self.tokenizer_g, "CLIP_G_tokens_2")

        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao gerar heatmap de atenção: {e}")
            import traceback
            traceback.print_exc()

    def _plot_and_save_heatmap(self, step: int, image_tag: str, aggregated_heatmap_data: np.ndarray, 
                              token_ids: torch.Tensor, tokenizer: object, output_suffix: str):
        """
        Função helper robusta que plota e salva um grid de heatmaps para um conjunto de tokens.
        """
        # 1. Decodifica e Filtra os Tokens
        tokens_text_raw = [tokenizer.decode(tid) for tid in token_ids]
        
        clean_tokens = []
        clean_indices = []
        # Ignora o primeiro token (<startoftext>) e para no primeiro padding (<endoftext>)
        for i, token in enumerate(tokens_text_raw[1:], start=1):
            if token == tokenizer.eos_token or (hasattr(tokenizer, 'pad_token') and token == tokenizer.pad_token):
                break
            clean_tokens.append(token.replace('</w>', '').strip())
            clean_indices.append(i)

        if not clean_tokens:
            print(f"[Visualizer] Nenhum token válido para plotar para {output_suffix}.")
            return

        # 2. Pega os dados de heatmap correspondentes aos tokens limpos
        heatmaps_to_plot = aggregated_heatmap_data[clean_indices, :, :]
        num_tokens_to_plot = len(clean_tokens)

        # 3. Calcula o grid para mostrar TUDO
        cols = 8  # 8 imagens por linha é um bom padrão
        rows = (num_tokens_to_plot + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=120)
        fig.suptitle(f'Attention Heatmaps ({output_suffix}) - Step {step}', fontsize=20)
        

        # Garante que `axes` seja sempre um array para fácil iteração
        if num_tokens_to_plot <= 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flat

        # 4. Plota cada heatmap
        for i in range(len(axes_flat)):
            ax = axes_flat[i]
            if i < num_tokens_to_plot:
                heatmap = heatmaps_to_plot[i, :, :]
                ax.imshow(heatmap, cmap='viridis')
                ax.set_title(f'"{clean_tokens[i]}"', fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off') # Esconde eixos de subplots não utilizados

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 5. Salva como JPEG

        outfile = self.out_dir / f"{image_tag}_HEATMAP_step_{step:06d}_{output_suffix}.jpg"
        plt.savefig(outfile, format='jpeg', dpi=96, pil_kwargs={'quality': 80})
        plt.close(fig)
        #print(f"[TokenAnalyzer] Heatmap ({output_suffix}) salvo em: {outfile}")

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