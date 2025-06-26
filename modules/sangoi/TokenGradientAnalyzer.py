import os
import pathlib
import re
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from collections import defaultdict

class TokenGradientAnalyzer:
    """
    Calcula a "afinidade" de tokens e tags em relação ao batch de imagens.
    A afinidade indica quais tokens/tags, se presentes, teriam maior
    probabilidade de reduzir a perda. Inclui ranking de booru tags e
    cálculo de EMA, Mediana e MAD para as afinidades.
    """

    def __init__(
        self,
        tokenizer_l,
        tokenizer_g,
        top_k_tokens: int = 5000, # Reduzido para relatórios de step mais enxutos
        out_dir: str = "token_affinity_reports",
        *,
        tag_file: Optional[str] = None,
        tag_cache_file: str = "tag_embedding_cache.pt",
        top_n_tags: int = 100, # Reduzido para relatórios de step mais enxutos
        ema_beta: float = 0.9,
        report_global_stats_top_n: int = 200, # Quantos itens mostrar no relatório global de EMA/Mediana/MAD
    ):
        if not (0 < ema_beta < 1):
            raise ValueError("ema_beta deve estar entre 0 e 1")

        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g
        self.top_k_tokens = top_k_tokens # Para relatórios de step (score atual + EMA)
        self.out_dir = out_dir
        self.report_global_stats_top_n = report_global_stats_top_n # Para o relatório global periódico

        self.tag_file = pathlib.Path(tag_file) if tag_file else None
        self.tag_cache_file = pathlib.Path(tag_cache_file)
        self.top_n_tags = top_n_tags # Para relatórios de step (score atual + EMA)
        self._tags_text_list: List[str] = []
        self._tag_embeddings_matrix: Optional[torch.Tensor] = None

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._grad_l: Optional[torch.Tensor] = None
        self._grad_g: Optional[torch.Tensor] = None
        self.armed = False
        self.pending_data: Dict[str, Any] = {}

        # --- Métricas de Afinidade ---
        self.ema_beta = ema_beta
        # EMA
        self.token_ema_scores: Dict[int, float] = {}
        self.tag_ema_scores: Dict[str, float] = {}
        # Histórico para Mediana/MAD (ATENÇÃO: Custo de Memória)
        self.token_all_scores_history: Dict[int, List[float]] = defaultdict(list)
        self.tag_all_scores_history: Dict[str, List[float]] = defaultdict(list)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TokenAnalyzer] Inicializado. Usando device: {self.device}")

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model

    def _backward_hook_l(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self._grad_l = grad_output[0].detach()

    def _backward_hook_g(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self._grad_g = grad_output[0].detach()

    def register_hooks(self, model: torch.nn.Module):
        self.remove_hooks()
        if model is None:
            print("[TokenAnalyzer] ERRO CRÍTICO: Modelo fornecido para register_hooks é None. Hooks não registrados.")
            return
        unwrapped_model = self._unwrap_model(model)
        if unwrapped_model is None:
            print("[TokenAnalyzer] ERRO CRÍTICO: Modelo 'unwrapped' é None em register_hooks. Hooks não registrados.")
            return
        if not hasattr(unwrapped_model, 'text_encoder_1') or unwrapped_model.text_encoder_1 is None:
            print("[TokenAnalyzer] ERRO: unwrapped_model.text_encoder_1 não existe ou é None. Hook para L não registrado.")
            return
        if not hasattr(unwrapped_model, 'text_encoder_2') or unwrapped_model.text_encoder_2 is None:
            print("[TokenAnalyzer] ERRO: unwrapped_model.text_encoder_2 não existe ou é None. Hook para G não registrado.")
            return
        try:
            handle_l = unwrapped_model.text_encoder_1.text_model.embeddings.register_full_backward_hook(self._backward_hook_l)
            self._hook_handles.append(handle_l)
            handle_g = unwrapped_model.text_encoder_2.text_model.embeddings.register_full_backward_hook(self._backward_hook_g)
            self._hook_handles.append(handle_g)
            print("[TokenAnalyzer] Hooks registrados nos Text Encoders L e G.")
        except AttributeError as e:
            print(f"[TokenAnalyzer] ERRO CRÍTICO ao registrar hooks: {e}.")
            self.remove_hooks()

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def set_pending_analysis(self, step: int, batch: Dict[str, Any]):
        self._grad_l = None
        self._grad_g = None
        self.pending_data = {"step": step, "batch": batch}
        self.armed = True

    def is_armed(self) -> bool:
        return self.armed

    def analyze_and_save_report(self, model: torch.nn.Module):
        if not self.is_armed():
            return

        step = self.pending_data.get("step", -1)
        batch = self.pending_data.get("batch", {})
        self.armed = False 

        try:
            if self._grad_l is None or self._grad_g is None:
                print(f"[TokenAnalyzer] Aviso: Gradientes não capturados no step {step}. Pulando análise.", flush=True)
                return
            if model is None:
                print(f"[TokenAnalyzer] ERRO CRÍTICO: Modelo é None. Step {step}. Abortando.")
                return
            unwrapped_model_base = self._unwrap_model(model)
            if unwrapped_model_base is None:
                print(f"[TokenAnalyzer] ERRO CRÍTICO: Modelo 'unwrapped' é None. Step {step}. Abortando.")
                return
            
            current_model_for_analysis = unwrapped_model_base
            try:
                candidate_model_on_device = unwrapped_model_base.to(self.device)
                current_model_for_analysis = candidate_model_on_device
            except Exception as e:
                print(f"[TokenAnalyzer] AVISO (Step {step}): Exceção ao mover modelo: {e}. Usando original.")

            grad_l_mean = -self._grad_l.mean(dim=(0, 1))
            grad_g_mean = -self._grad_g.mean(dim=(0, 1))

            with torch.no_grad():
                self._analyze_tokens(current_model_for_analysis, grad_l_mean, grad_g_mean, step, batch)
                if self.tag_file and self.tag_file.exists():
                    self._analyze_booru_tags(current_model_for_analysis, grad_l_mean, grad_g_mean, step, batch)
                elif self.tag_file:
                    print(f"[TokenAnalyzer] Aviso: Tag file '{self.tag_file}' não encontrado (Step {step}).", flush=True)

        except Exception as e:
            print(f"[TokenAnalyzer] ERRO CRÍTICO INESPERADO (Step {step}): {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            self._grad_l = None 
            self._grad_g = None
            self.pending_data = {}

    def _analyze_tokens(
        self, current_model_obj: torch.nn.Module, grad_l_mean: torch.Tensor, grad_g_mean: torch.Tensor,
        step: int, batch: Dict[str, Any]
    ):
        affinity_l, affinity_g = torch.tensor(0.0), torch.tensor(0.0) # Init
        vocab_size_l = self.tokenizer_l.vocab_size if hasattr(self.tokenizer_l, 'vocab_size') else 32000
        vocab_size_g = self.tokenizer_g.vocab_size if hasattr(self.tokenizer_g, 'vocab_size') else 32000
        
        # Encoder L
        if hasattr(current_model_obj, 'text_encoder_1') and current_model_obj.text_encoder_1 is not None:
            try:
                emb_matrix_l = current_model_obj.text_encoder_1.get_input_embeddings().weight.to(grad_l_mean.device)
                affinity_l = F.cosine_similarity(grad_l_mean, emb_matrix_l, dim=-1)
            except Exception as e:
                print(f"[TokenAnalyzer] ERRO (Step {step}) TE1: {e}")
                affinity_l = torch.zeros(vocab_size_l, device=grad_l_mean.device)
        else:
            print(f"[TokenAnalyzer] ERRO (Step {step}): TE1 ausente/None.")
            affinity_l = torch.zeros(vocab_size_l, device=grad_l_mean.device)

        # Encoder G
        if hasattr(current_model_obj, 'text_encoder_2') and current_model_obj.text_encoder_2 is not None:
            try:
                emb_matrix_g = current_model_obj.text_encoder_2.get_input_embeddings().weight.to(grad_g_mean.device)
                affinity_g = F.cosine_similarity(grad_g_mean, emb_matrix_g, dim=-1)
            except Exception as e:
                print(f"[TokenAnalyzer] ERRO (Step {step}) TE2: {e}")
                affinity_g = torch.zeros(vocab_size_g, device=grad_g_mean.device)
        else:
            print(f"[TokenAnalyzer] ERRO (Step {step}): TE2 ausente/None.")
            affinity_g = torch.zeros(vocab_size_g, device=grad_g_mean.device)
        
        def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
            min_s, max_s = scores.min(), scores.max()
            denom = max_s - min_s
            return (scores - min_s) / (denom + 1e-8) if denom > 1e-8 else torch.zeros_like(scores)

        norm_affinity_l = _normalize_scores(affinity_l)
        norm_affinity_g = _normalize_scores(affinity_g)

        if norm_affinity_l.shape == norm_affinity_g.shape:
            total_affinity = norm_affinity_l + norm_affinity_g
        else:
            print(f"[TokenAnalyzer] AVISO (Step {step}): Vocabulários L e G com tamanhos diferentes. Usando apenas G para afinidade de tokens.")
            total_affinity = norm_affinity_g # Ou o maior deles

        if total_affinity.max() == 0 and total_affinity.min() == 0:
             print(f"[TokenAnalyzer] AVISO (Step {step}): Afinidade total dos tokens é zero.")
             sorted_indices = torch.arange(len(total_affinity), device=total_affinity.device)
        else:
            sorted_indices = torch.argsort(total_affinity, descending=True)
        
        self._update_all_score_metrics(
            current_scores=total_affinity, 
            is_token=True
        )
        self._save_token_report(step, batch, sorted_indices, total_affinity, self.token_ema_scores)

    def _analyze_booru_tags(
        self, current_model_obj: torch.nn.Module, grad_l_mean: torch.Tensor, grad_g_mean: torch.Tensor,
        step: int, batch: Dict[str, Any]
    ):
        if not hasattr(current_model_obj, 'text_encoder_2') or current_model_obj.text_encoder_2 is None:
            print(f"[TokenAnalyzer] ERRO (Step {step}): TE2 ausente para Booru tags.")
            return

        tag_list, tag_embeddings = self._get_or_create_tag_embeddings(current_model_obj)
        if tag_embeddings is None or not tag_list:
            print(f"[TokenAnalyzer] Sem embeddings de tags (Step {step}).", flush=True)
            return

        tag_embeddings = tag_embeddings.to(grad_g_mean.device) 
        grad_ref_tags = F.normalize(grad_g_mean, dim=0)
        tag_affinities = torch.matmul(tag_embeddings, grad_ref_tags)
        
        if tag_affinities.numel() == 0:
            print(f"[TokenAnalyzer] AVISO (Step {step}): Afinidades de tags vazias.")
            return

        sorted_tag_indices = torch.argsort(tag_affinities, descending=True)
        
        # Atualiza EMA e histórico de scores
        self._update_all_score_metrics(
            current_scores=tag_affinities, 
            is_token=False, 
            item_list_for_keys=tag_list
        )
        self._save_tag_report(step, batch, tag_list, sorted_tag_indices, tag_affinities, self.tag_ema_scores)

    def _get_or_create_tag_embeddings(self, model_obj_for_emb: torch.nn.Module) -> Tuple[List[str], Optional[torch.Tensor]]:
        if self._tag_embeddings_matrix is not None and self._tags_text_list:
            return self._tags_text_list, self._tag_embeddings_matrix.to(torch.float32)
        if self.tag_cache_file.exists():
            try:
                cached_data = torch.load(self.tag_cache_file, map_location="cpu")
                self._tags_text_list = cached_data["tags"]
                self._tag_embeddings_matrix = cached_data["embeddings_matrix"]
                return self._tags_text_list, self._tag_embeddings_matrix.to(torch.float32)
            except Exception as e:
                print(f"[TokenAnalyzer] Falha ao carregar cache de tags: {e}. Gerando.", flush=True)
        if not self.tag_file or not self.tag_file.exists(): return [], None
        print(f"[TokenAnalyzer] Gerando embeddings para tags de '{self.tag_file}'...")
        tags = sorted(list(set(t.strip() for t in self.tag_file.read_text(encoding="utf-8").splitlines() if t.strip())))
        if not tags: print("[TokenAnalyzer] Nenhuma tag encontrada."); return [], None
        if not hasattr(model_obj_for_emb, 'text_encoder_2') or model_obj_for_emb.text_encoder_2 is None:
            print("[TokenAnalyzer] ERRO: TE2 não encontrado para gerar embeddings de tags.")
            return [], None
        try:
            text_encoder_g_module = model_obj_for_emb.text_encoder_2.eval().to(self.device)
            tokenizer_g_ref = self.tokenizer_g
        except Exception as e:
             print(f"[TokenAnalyzer] ERRO ao preparar TE2 para cache de tags: {e}."); return [], None
        batch_size = 128 
        all_tag_embeddings_list: List[torch.Tensor] = []
        for i in range(0, len(tags), batch_size):
            batch_tags = tags[i:i+batch_size]
            tokens_g = tokenizer_g_ref(batch_tags, padding="max_length", max_length=tokenizer_g_ref.model_max_length, 
                                       truncation=True, return_tensors="pt", add_special_tokens=False).to(self.device)
            with torch.no_grad():
                embeddings_g = text_encoder_g_module.get_input_embeddings()(tokens_g.input_ids)
                mask_g = tokens_g.attention_mask.unsqueeze(-1).expand_as(embeddings_g)
                summed_embeddings_g = (embeddings_g * mask_g).sum(dim=1)
                num_valid_tokens = torch.max(mask_g.sum(dim=1), torch.ones_like(mask_g.sum(dim=1)))
                mean_embeddings_g = summed_embeddings_g / num_valid_tokens
                all_tag_embeddings_list.append(mean_embeddings_g.cpu())
        if not all_tag_embeddings_list: print("[TokenAnalyzer] Nenhum embedding de tag gerado."); return [], None
        tag_embeddings_matrix = F.normalize(torch.cat(all_tag_embeddings_list, dim=0), p=2, dim=1)
        self._tags_text_list = tags
        self._tag_embeddings_matrix = tag_embeddings_matrix.to(torch.float16)
        try:
            self.tag_cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"tags": self._tags_text_list, "embeddings_matrix": self._tag_embeddings_matrix}, self.tag_cache_file)
            print(f"[TokenAnalyzer] Cache de tags salvo em '{self.tag_cache_file}'.")
        except Exception as e:
            print(f"[TokenAnalyzer] Falha ao salvar cache de tags: {e}", flush=True)
        return self._tags_text_list, self._tag_embeddings_matrix.to(torch.float32)

    def _update_all_score_metrics(
        self,
        current_scores: torch.Tensor, # Tensor 1D com os scores atuais para todos os itens
        is_token: bool,
        item_list_for_keys: Optional[List[str]] = None # Apenas para tags (self._tags_text_list)
    ):
        """Atualiza EMA e armazena o score atual no histórico para Mediana/MAD."""
        
        num_items = len(current_scores)
        
        for i in range(num_items):
            current_score_val = current_scores[i].item()
            
            if is_token:
                key = i # ID do token
                ema_dict = self.token_ema_scores
                history_dict = self.token_all_scores_history
            else: # Tags
                if item_list_for_keys is None or i >= len(item_list_for_keys):
                    # print(f"[TokenAnalyzer] AVISO: Índice {i} fora dos limites para item_list_for_keys em _update_all_score_metrics para tags.")
                    continue 
                key = item_list_for_keys[i] # Texto da tag
                ema_dict = self.tag_ema_scores
                history_dict = self.tag_all_scores_history

            # Atualizar EMA
            old_ema_score = ema_dict.get(key, current_score_val) # Inicializa com o score atual se não existir
            ema_dict[key] = self.ema_beta * old_ema_score + (1 - self.ema_beta) * current_score_val
            
            # Adicionar ao histórico para Mediana/MAD
            history_dict[key].append(current_score_val)

    def _calculate_median_mad(self, scores_history_list: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """Calcula Mediana e MAD para uma lista de scores."""
        if not scores_history_list or len(scores_history_list) < 2: # Precisa de pelo menos 2 pontos para MAD
            return None, None

        scores_tensor = torch.tensor(scores_history_list, dtype=torch.float32)
        
        median_val = torch.median(scores_tensor).item()
        
        # MAD = median(|score_i - median(scores)|)
        abs_deviations = torch.abs(scores_tensor - median_val)
        mad_val = torch.median(abs_deviations).item()
        
        return median_val, mad_val
            
    @staticmethod
    def _sanitize_filename(candidate: str) -> str:
        basename = os.path.splitext(os.path.basename(candidate))[0]
        return re.sub(r"[^\w.\-]+", "_", basename)

    def _get_common_report_header(self, step: int, batch: Dict[str, Any], report_type: str = "Afinidade") -> Tuple[List[str], str]:
        image_paths_val = batch.get("image_path", batch.get("image_paths", ["N/A"]))
        if not isinstance(image_paths_val, list): image_paths_val = [str(image_paths_val)]
        image_tag = self._sanitize_filename(image_paths_val[0]) if image_paths_val and image_paths_val[0] not in ["N/A", ""] else f"batch_step_{step}"
        
        report_header = [
            f"### Relatório de {report_type} - Step: {step}\n",
            f"### Imagem(ns): {', '.join(image_paths_val)}\n",
            f"### EMA Beta: {self.ema_beta}\n",
            "### NOTA: Afinidade alta indica que o token/tag provavelmente reduziria a perda.\n",
            "-------------------------------------------------\n\n"
        ]
        return report_header, image_tag

    def _format_ema_report_lines( # Mantido para relatórios de step
            self, ema_scores_dict: Dict[Any, float],
            is_token: bool, top_n: int
        ) -> List[str]:
        if not ema_scores_dict: return ["Nenhuma pontuação EMA para reportar.\n"]
        sorted_ema_items = sorted(ema_scores_dict.items(), key=lambda item: item[1], reverse=True)
        lines = [f"--- TOP {min(top_n, len(sorted_ema_items))} ITENS POR SCORE EMA ---\n"]
        header = "Score EMA  | Token (ID)\n" if is_token else "Score EMA  | Tag\n"
        lines.append(header)
        lines.append("-----------|----------------------------------\n")
        for item_key, ema_score in sorted_ema_items[:top_n]:
            text = self._decode_token_id(int(item_key)) if is_token else str(item_key)
            id_suffix = f" (ID: {item_key})" if is_token else ""
            lines.append(f"{ema_score:10.6f} | {text}{id_suffix}\n")
        lines.append("\n")
        return lines

    def _decode_token_id(self, token_id: int) -> str:
        # (Implementação anterior mantida, pode ser otimizada se necessário)
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

    # Relatórios de Step (Score Atual + EMA) - Mantidos para feedback rápido
    def _save_token_report(
        self, step: int, batch: Dict[str, Any],
        sorted_indices: torch.Tensor, current_scores: torch.Tensor,
        token_ema_scores: Dict[int, float] # Passando o dict de EMA
    ):
        report_header, image_tag = self._get_common_report_header(step, batch, report_type="Afinidade de Tokens (Step)")
        out_dir = pathlib.Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outfile = out_dir / f"step_{step:06d}_{image_tag}_tokens.txt"
        report_lines = list(report_header)
        report_lines.append(f"--- TOP {min(self.top_k_tokens, len(sorted_indices))} TOKENS (Score Atual) ---\n")
        report_lines.append("Score Atual| Token (ID)\n")
        report_lines.append("-----------|----------------------------------\n")
        for i in range(min(self.top_k_tokens, len(sorted_indices))):
            token_id = sorted_indices[i].item()
            score = current_scores[token_id].item() if token_id < len(current_scores) else 0.0
            token_text = self._decode_token_id(token_id)
            report_lines.append(f"{score:10.6f} | {token_text} (ID: {token_id})\n")
        report_lines.append("\n")
        report_lines.extend(self._format_ema_report_lines(token_ema_scores, is_token=True, top_n=self.top_k_tokens)) # Usar top_k_tokens para EMA no report de step
        try:
            outfile.write_text("".join(report_lines), encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao salvar relatório de tokens (Step {step}): {e}")

    def _save_tag_report(
        self, step: int, batch: Dict[str, Any],
        tag_list: List[str], sorted_tag_indices: torch.Tensor, current_tag_scores: torch.Tensor,
        tag_ema_scores: Dict[str, float] # Passando o dict de EMA
    ):
        report_header, image_tag = self._get_common_report_header(step, batch, report_type="Afinidade de Booru Tags (Step)")
        out_dir = pathlib.Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outfile = out_dir / f"step_{step:06d}_{image_tag}_booru_tags.txt"
        report_lines = list(report_header)
        report_lines.append(f"--- TOP {min(self.top_n_tags, len(sorted_tag_indices))} BOORU TAGS (Score Atual) ---\n")
        report_lines.append("Score Atual| Tag\n")
        report_lines.append("-----------|----------------------------------\n")
        for i in range(min(self.top_n_tags, len(sorted_tag_indices))):
            original_idx = sorted_tag_indices[i].item()
            if original_idx < len(tag_list) and original_idx < len(current_tag_scores):
                tag_text = tag_list[original_idx]
                score = current_tag_scores[original_idx].item()
                report_lines.append(f"{score:10.6f} | {tag_text}\n")
        report_lines.append("\n")
        report_lines.extend(self._format_ema_report_lines(tag_ema_scores, is_token=False, top_n=self.top_n_tags)) # Usar top_n_tags para EMA no report de step
        try:
            outfile.write_text("".join(report_lines), encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao salvar relatório de tags (Step {step}): {e}")

    # NOVO: Relatório Global Periódico com EMA, Mediana e MAD
    def save_global_stats_report(self, current_step_or_epoch: int, is_epoch_end: bool = False):
        """
        Salva um relatório global com EMA, Mediana e MAD para os tokens/tags mais relevantes.
        Chame periodicamente (ex: final de epoch ou a cada N steps).

        Args:
            current_step_or_epoch (int): O step ou epoch atual para nomear o arquivo.
            is_epoch_end (bool): Se True, nomeia o arquivo como epoch_XXX, senão step_XXX.
        """
        out_dir = pathlib.Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        file_prefix = "epoch" if is_epoch_end else "step"

        # --- Relatório Global de Tokens ---
        if self.token_ema_scores:
            outfile_tokens = out_dir / f"{file_prefix}_{current_step_or_epoch:06d}_GLOBAL_STATS_tokens.txt"
            report_lines = [
                f"### Relatório Global de Estatísticas de Tokens - {'Epoch' if is_epoch_end else 'Step'}: {current_step_or_epoch}\n",
                f"### EMA Beta: {self.ema_beta}\n",
                f"### Top N Reportado: {self.report_global_stats_top_n}\n",
                "-------------------------------------------------\n\n",
                f"--- TOP {self.report_global_stats_top_n} TOKENS (Ordenado por EMA Descendente) ---\n",
                "EMA        | Mediana    | MAD        | Token (ID)\n",
                "-----------|------------|------------|----------------------------------\n",
            ]

            # Ordenar por EMA para pegar os mais relevantes
            sorted_tokens_by_ema = sorted(self.token_ema_scores.items(), key=lambda item: item[1], reverse=True)
            
            for token_id, ema_score in sorted_tokens_by_ema[:self.report_global_stats_top_n]:
                history = self.token_all_scores_history.get(token_id, [])
                median_val, mad_val = self._calculate_median_mad(history)
                
                median_str = f"{median_val:10.6f}" if median_val is not None else "N/A       "
                mad_str = f"{mad_val:10.6f}" if mad_val is not None else "N/A       "
                token_text = self._decode_token_id(token_id)
                
                report_lines.append(f"{ema_score:10.6f} | {median_str} | {mad_str} | {token_text} (ID: {token_id})\n")
            
            try:
                outfile_tokens.write_text("".join(report_lines), encoding="utf-8", errors="ignore")
                print(f"[TokenAnalyzer] Relatório Global de Estatísticas de Tokens salvo em: {outfile_tokens}")
            except Exception as e:
                print(f"[TokenAnalyzer] ERRO ao salvar Relatório Global de Tokens (Step/Epoch {current_step_or_epoch}): {e}")

        # --- Relatório Global de Tags ---
        if self.tag_ema_scores and self.tag_file:
            outfile_tags = out_dir / f"{file_prefix}_{current_step_or_epoch:06d}_GLOBAL_STATS_booru_tags.txt"
            report_lines = [
                f"### Relatório Global de Estatísticas de Booru Tags - {'Epoch' if is_epoch_end else 'Step'}: {current_step_or_epoch}\n",
                f"### EMA Beta: {self.ema_beta}\n",
                f"### Top N Reportado: {self.report_global_stats_top_n}\n",
                "-------------------------------------------------\n\n",
                f"--- TOP {self.report_global_stats_top_n} BOORU TAGS (Ordenado por EMA Descendente) ---\n",
                "EMA        | Mediana    | MAD        | Tag\n",
                "-----------|------------|------------|----------------------------------\n",
            ]

            sorted_tags_by_ema = sorted(self.tag_ema_scores.items(), key=lambda item: item[1], reverse=True)

            for tag_text, ema_score in sorted_tags_by_ema[:self.report_global_stats_top_n]:
                history = self.tag_all_scores_history.get(tag_text, [])
                median_val, mad_val = self._calculate_median_mad(history)

                median_str = f"{median_val:10.6f}" if median_val is not None else "N/A       "
                mad_str = f"{mad_val:10.6f}" if mad_val is not None else "N/A       "
                
                report_lines.append(f"{ema_score:10.6f} | {median_str} | {mad_str} | {tag_text}\n")

            try:
                outfile_tags.write_text("".join(report_lines), encoding="utf-8", errors="ignore")
                print(f"[TokenAnalyzer] Relatório Global de Estatísticas de Booru Tags salvo em: {outfile_tags}")
            except Exception as e:
                print(f"[TokenAnalyzer] ERRO ao salvar Relatório Global de Tags (Step/Epoch {current_step_or_epoch}): {e}")