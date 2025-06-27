import os
import pathlib
import re
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
from collections import defaultdict

# Importa as ferramentas que você (finalmente) concordou em usar.
from modules.modelSetup.BaseStableDiffusionXLSetup import AttentionMapLogger, CapturingAttnProcessor

class TokenGradientAnalyzer:
    """
    Calcula afinidade de tokens via gradiente E atenção.
    A análise de atenção agora é robusta e desacoplada, usando um sistema de
    substituição de processador em vez de hooks frágeis.
    """

    # ========================================================================= #
    # Construtor e Estado
    # ========================================================================= #
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
            self._original_attn_processors = model.unet.attn_processors.copy()
            capturing_processor = CapturingAttnProcessor(self.map_logger)
            
            # Substitui apenas os processadores de cross-attention
            processors_to_capture = {name for name in self._original_attn_processors if "attn2" in name}
            for name in processors_to_capture:
                # NÃO É SET_ATTN_PROCESSOR.
                # É SET_PROCESSOR.
                # SET. PROCESSOR.
                model.unet.set_processor(name, capturing_processor)
            
            self.is_capturing_attention = True
            print(f"[TokenAnalyzer] {len(processors_to_capture)} processadores de atenção substituídos.")

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

    # ========================================================================= #
    # LÓGICA DE ANÁLISE (Onde os dados são consumidos)
    # ========================================================================= #

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
        attn_maps = self.map_logger.get_maps()
        if not attn_maps:
            # Isso não é um erro, pode acontecer se requires_grad=False (ex: passo incondicional)
            return

        # Agrega os mapas de todas as camadas de cross-attention capturadas
        total_map = torch.stack(attn_maps).sum(dim=0)  # Shape: (B, Q, K)
        # Soma a atenção recebida por cada token de texto (dim K) em todas as queries espaciais (dim Q)
        # e faz a média no batch.
        attn_scores = total_map.sum(dim=1).mean(dim=0) # Shape: (K,)

        self._update_all_score_metrics(attn_scores, is_token=True, use_attention_scores=True)
        self._save_report(step, batch, attn_scores, "Attention Tokens", self.attn_token_ema_scores)

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
        outfile = self.out_dir / f"step_{step:06d}_{image_tag}_{safe_report_type}.txt"

        # Prepara as linhas do relatório
        report_lines = list(report_header)
        report_lines.append(f"--- TOP {self.top_k_tokens} TOKENS (Score Atual) ---\n")
        report_lines.extend(self._format_report_lines(current_scores, is_token=True, top_n=self.top_k_tokens, score_type="Atual"))
        report_lines.append("\n")
        report_lines.append(f"--- TOP {self.top_k_tokens} TOKENS (Score EMA Acumulado) ---\n")
        report_lines.extend(self._format_report_lines(ema_scores, is_token=True, top_n=self.top_k_tokens, score_type="EMA"))

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
    