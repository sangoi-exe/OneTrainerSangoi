from collections import defaultdict
import re
import os
import torch

import pathlib
import numpy as np
import torch.nn.functional as F
import matplotlib

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.config.TrainConfig import TrainConfig

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.util.time_util import get_string_timestamp
from typing import Dict, Any, List, Optional, Tuple, cast

# Assumindo que estas classes estão em um módulo acessível, como você definiu.
from transformers import CLIPTokenizer
from diffusers.models.attention_processor import Attention
from modules.modelSetup.BaseStableDiffusionXLSetup import AttentionMapLogger, CaptureManager

from torch import Tensor


class CrossAttnMapsAnalyzer:
    """
    Calcula afinidade de tokens via gradiente e/ou atenção.
    Usa um sistema de captura de atenção robusto e desacoplado.
    """

    def __init__(
        self,
        drop_head_mask,
        config: TrainConfig,
        model: StableDiffusionXLModel,
        tokenizer_l: CLIPTokenizer,
        tokenizer_g: CLIPTokenizer,
        out_dir: str = "token_affinity_reports",
    ):
        """
        Construtor simplificado e focado no essencial.
        """
        # --- Configurações Essenciais ---
        self.model = model
        self.analyzer_interval = config.analyzer_interval
        self.enable_attn_report = config.analyzer_enable_attn_report
        self.enable_grad_report = config.analyzer_enable_grad_report
        self.enable_heatmap_report = config.analyzer_enable_heatmaps
        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g
        self.drop_head_mask = drop_head_mask

        # --- Ferramentas e Estado ---
        self.map_logger = AttentionMapLogger()
        self.cap_manager = CaptureManager(logger=self.map_logger, drop_mask_ref=self.drop_head_mask)
        
        self._grad_l: Optional[Tensor] = None
        self._grad_g: Optional[Tensor] = None
        self.pending_data: Dict[str, Any] = {}
        
        self.strikes = defaultdict(int)
        self.reinits = defaultdict(int)
        
        # --- Setup do Diretório de Saída ---
        base_out_dir = pathlib.Path(out_dir)
        timestamp = get_string_timestamp(style='file')
        self.out_dir = base_out_dir / timestamp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TokenAnalyzer] Relatórios serão salvos em: {self.out_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TokenAnalyzer] Inicializado. Device: {self.device}")

        self.templates: Dict[str, Dict[str, Any]] = {}

        self.alias2path = {}
        self.layer_aliases = []
        for path, m in self.model.unet.named_modules():
            if isinstance(m, Attention) and "attn2" in path:
                alias = path.replace('.', '_') # padroniza
                self.alias2path[alias] = path
                self.layer_aliases.append(alias) # ordem idêntica à captura
                
    def path_to_alias(path: str) -> str:
        return path.replace('.', '_')
    
    def update_strike_count(self, per_head_entropy, thr: float = 3.3, patience: int = 3):
        """
        Atualiza strikes e devolve lista de heads que bateram o limite.
        - thr      : entropia abaixo deste valor conta strike.
        - patience : nº de strikes antes de tentar reinit.
        """
        killers = self.drop_head_mask
        for tag, val in per_head_entropy.items():
            if killers.get(tag):
                continue
            # zera strike se a entropia subir; soma se continuar baixa
            self.strikes[tag] = self.strikes[tag] + 1 if val < thr else 0

        # devolve quem excedeu patience
        return [t for t, s in self.strikes.items() if s >= patience]

    def _reinit_heads(self, dead_tags, std: float = 0.01):
        """
        Reinicializa ou guilhotina heads.
        - tag formato: "<alias>_hXX" (alias = path com '_' )
        """
        for tag in dead_tags:
            # 1. Se já reiniciei 3 vezes, dropa de vez
            if self.reinits[tag] >= 3:
                self.drop_head_mask[tag] = True
                self.strikes.pop(tag, None)
                continue

            # 2. Resolve alias → módulo
            try:
                alias, hstr = tag.rsplit('_h', 1)
                head_idx = int(hstr)
                path = self.alias2path[alias]
                mod = self.model.unet.get_submodule(path)
            except (ValueError, KeyError, AttributeError):
                print(f"[WARN] Tag inválida ou módulo inexistente: {tag}")
                continue
            if not isinstance(mod, Attention):
                print(f"[WARN] {path} não é Attention. Pulando.")
                continue

            # 3. Slice dos parâmetros da cabeça
            heads  = getattr(mod, "num_heads", getattr(mod, "heads"))
            head_dim = mod.to_q.weight.shape[0] // heads
            s, e = head_idx * head_dim, (head_idx + 1) * head_dim

            for proj in (mod.to_q, mod.to_k, mod.to_v, mod.to_out[0]):
                with torch.no_grad():
                    torch.nn.init.normal_(proj.weight.data[s:e], 0.0, std)
                    if proj.bias is not None:
                        torch.nn.init.zeros_(proj.bias.data[s:e])

            # 4. Livro-caixa
            self.reinits[tag] += 1
            self.strikes[tag] = 0
            print(f"[REINIT] {tag} reiniciada ({self.reinits[tag]}×).")


    def set_pending_analysis(self, step: int, batch: Dict[str, Any]):
        """Prepara para um novo passo de análise. Limpa o estado antigo."""
        self._grad_l, self._grad_g = None, None
        # O logger é limpo pelo AttnCaptureManager, mas uma limpeza extra aqui não faz mal.
        self.map_logger.clear()
        self.pending_data = {"step": step, "batch": batch}

    def analyze_gradients_after_backward(self, model: StableDiffusionXLModel):
        """Chamado após loss.backward() para analisar gradientes."""
        if not self.enable_grad_report:
            return

        step = self.pending_data.get("step", -1)
        batch = self.pending_data.get("batch", {})
        if not batch: return

        try:
            self._analyze_tokens_from_gradient(step, batch)
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
            analyzer_interval = self.analyzer_interval > 0 and (current_epoch % self.analyzer_interval == 0)
            if analyzer_interval:
                self._visualize_attention_heatmaps(attn_maps_to_analyze, step, batch)
        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao analisar atenção/heatmaps (Step {step}): {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.pending_data.clear()

    def _analyze_tokens_from_gradient(self, step: int, batch: Dict[str, Any]):
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

    def _visualize_attention_heatmaps(self, attn_maps_raw: List[Tensor], step: int, batch: Dict[str, Any]):
        """
        Orquestrador principal: agrega os mapas e chama a função de plotagem
        para ambos os conjuntos de tokens (CLIP-L e CLIP-G).
        """

        if not attn_maps_raw:
            print(f"[TokenAnalyzer] SEM ATTENTION MAP PRO HEATMAP")
            return
        try:
            base_batch_size: int = batch['tokens_1'].shape[0]
            conditional_maps: List[Tensor] = [m[:base_batch_size] for m in attn_maps_raw]
            
            latent: Tensor = cast(Tensor, batch['latent_image'])
            latent_h: int = int(latent.shape[2])
            latent_w: int = int(latent.shape[3])

            aspect_ratio = latent_h / latent_w if latent_w > 0 else 1.0

            min_q_dim = min(m.shape[2] for m in conditional_maps)
            target_h, target_w = self._infer_spatial_dims(min_q_dim, aspect_ratio)

            if target_h == -1: return

            normalized_maps: List[Tensor] = []
            for cond_map in conditional_maps:
                map_avg_heads: Tensor = cond_map.mean(dim=1)
                current_q_dim: int = map_avg_heads.shape[1]
                current_h, current_w = self._infer_spatial_dims(current_q_dim, aspect_ratio)
                if current_h == -1: continue
                
                num_tokens = map_avg_heads.shape[2]
                map_reshaped = map_avg_heads.permute(0, 2, 1).view(1, num_tokens, current_h, current_w)
                map_resized = F.interpolate(map_reshaped, size=(target_h, target_w), mode='bilinear', align_corners=False)
                normalized_maps.append(map_resized)

            if not normalized_maps: return

            aggregated_heatmap_data = torch.stack(normalized_maps).mean(dim=0).squeeze(0).to(torch.float32).cpu().numpy()

            # --- ETAPA 2: LÓGICA DE GABARITO E ORDENAÇÃO ---
            concept_name = batch.get("concept_name", "default")[0]
            img_key = batch.get("image_path", [""])[0]
            current_token_ids = batch['tokens_2'][0] # Usa CLIP-G

            template = self.templates.get(img_key)

            if concept_name == "orig" and template is None:
                self.templates[img_key] = {
                    "ids": current_token_ids.cpu().numpy(),
                    "txts": [self.tokenizer_g.decode(t) for t in current_token_ids],
                }
                template = self.templates[img_key]
                print(f"[ANALYSIS] Gabarito salvo para '{img_key}' (step {step}).")

            # Se temos um gabarito, SEMPRE plotamos na ordem do gabarito.
            if template is not None and concept_name != "orig":
                # 1. Cria um mapa de posições para os tokens do prompt ATUAL.
                #    {token_id: [lista_de_indices_onde_ele_aparece]}
                #    Ex: {123: [2, 15], 456: [8]}
                current_positions = defaultdict(list)
                for idx, tid in enumerate(current_token_ids.tolist()):
                    current_positions[tid].append(idx)

                # 2. Constrói os dados ordenados para a plotagem.
                ordered_heatmap_data_list = []
                ordered_token_texts_list = []
                ordered_token_ids_list = []
                
                # Contador para saber qual ocorrência de um token repetido já usamos.
                # Ex: {123: 0} -> ainda não usamos nenhuma instância do token 123.
                usage_counter = defaultdict(int)

                # 3. Itera sobre o GABARITO para definir a ordem.
                for tid, txt in zip(template["ids"], template["txts"]):
                    
                    # Pega a contagem de uso para este ID de token
                    occurrence_index = usage_counter[tid]
                    
                    # Verifica se o prompt ATUAL tem essa ocorrência do token
                    if occurrence_index < len(current_positions[tid]):                        
                        # Se sim, pega o índice do heatmap correspondente no lote atual
                        current_heatmap_index = current_positions[tid][occurrence_index]                        
                        # Adiciona os dados na ordem correta
                        ordered_heatmap_data_list.append(aggregated_heatmap_data[current_heatmap_index])
                        ordered_token_texts_list.append(txt)
                        ordered_token_ids_list.append(tid)                        
                        # Incrementa o contador de uso para este ID
                        usage_counter[tid] += 1

                if not ordered_heatmap_data_list:
                    print(f"[WARN] Sem match entre template '{img_key}' e prompt atual.")
                    return

                # Converte as listas para os formatos corretos para a função de plotagem
                final_heatmap_data = np.stack(ordered_heatmap_data_list)
                final_token_ids = np.array(ordered_token_ids_list)
                final_token_texts = ordered_token_texts_list
                output_suffix = f"{concept_name}"
            else:
                # Se for o 'orig' ou se não houver gabarito, plota na ordem normal.
                final_heatmap_data = aggregated_heatmap_data
                final_token_ids = current_token_ids.cpu().numpy()
                final_token_texts = [self.tokenizer_g.decode(tid) for tid in current_token_ids]
                output_suffix = concept_name
                
            # Chama a função de plotagem com os dados devidamente ordenados e formatados
            _, image_tag = self._get_common_report_header(step, batch, "")
            self._plot_and_save_heatmap(
                step,
                image_tag,
                final_heatmap_data,
                final_token_ids,
                final_token_texts,
                output_suffix
            )

        except Exception as e:
            print(f"[TokenAnalyzer] ERRO ao gerar heatmap de atenção: {e}")
            import traceback
            traceback.print_exc()

    def _plot_and_save_heatmap(
            self,
            step: int,
            image_tag: str,
            heatmap_data: np.ndarray,
            token_ids: np.ndarray,
            token_texts: list,
            output_suffix: str,
            max_cols: int = 8,
        ):
        """
        Renderiza heatmaps sem matar pontuação,
        com grid adaptativo e títulos que não colidem.
        """

        SPECIAL = set(self.tokenizer_g.all_special_ids)

        # ---------- FILTRO ----------
        keep   = []
        labels = []
        for i, (tid, txt) in enumerate(zip(token_ids, token_texts)):
            if tid in SPECIAL: continue

            clean = txt.replace('</w>', '') # mantém vírgula, mantém espaço se houver
            clean = clean if clean != '' else ',' # token vazio aqui é vírgula pura
            labels.append(clean)
            keep.append(i)

        if not keep:
            return

        hm = heatmap_data[keep, :, :]
        n  = len(keep)

        # ---------- GRID ----------
        cols = min(max_cols, max(3, n))           # nunca menos que 3 colunas
        rows = (n + cols - 1) // cols

        h, w  = hm[0].shape
        ar    = h / w if w else 1.0

        # largura = 2 in por coluna; altura proporcional + margem p/ títulos
        fig_w = cols * 2
        fig_h = rows * (2 * ar) + 0.8

        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=200)
        fig.suptitle(f'Attention Heatmap ({output_suffix}) - Step {step}', fontsize=12)

        axes = axes.flat if isinstance(axes, np.ndarray) else [axes]

        # ---------- PLOT ----------
        for plot_i, real_i in enumerate(keep):
            ax = axes[plot_i]
            ax.imshow(hm[plot_i], cmap='viridis')
            ax.set_title(f'{real_i}: "{labels[plot_i]}"', fontsize=7, pad=4)
            ax.axis('off')

        # esconde vazios
        for j in range(n, len(axes)):
            axes[j].axis('off')

        plt.subplots_adjust(
            left=0.03, right=0.97,
            top=0.88 if rows == 1 else 0.90,
            bottom=0.04,
            wspace=0.08, hspace=0.25
        )

        out = self.out_dir / f"{image_tag}_HEATMAP_step_{step:06d}_{output_suffix}.jpg"
        plt.savefig(out, format='jpeg', dpi=96, pil_kwargs={'quality': 85})
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
        tokens_text = self.tokenizer_g.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        
        special = set(self.tokenizer_g.all_special_ids)
        scored_tokens = []
        for i, score in enumerate(scores):
            token_id = token_ids[i].item()
            token_text = tokens_text[i].replace('</w>', '').strip()
            
            if token_id in special: continue
            if token_text in ['<|endoftext|>', '</s>']: continue # seguro morreu de velho
            
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
        for alias, raw in zip(self.layer_aliases, attn_maps_raw):
            p = raw[:B].float().clamp_min_(1e-9)
            ent = -(p * p.log2()).sum(-1)        # (B,H,Q)
            ent_h = ent.mean(dim=(0, 2))         # (H,)
            for h, v in enumerate(ent_h):
                per_head[f"{alias}_h{h:02d}"] = v.item()
        return per_head
    
    def report_head_health(self, epoch: int | str):
        """
        Imprime no terminal um sumário do estado das heads
        ao final da epoch.

        • Guilhotinadas  → estão marcadas em drop_head_mask=True
        • Strikes ativos → strikes[tag] > 0 e ainda não guilhotinadas
        """
        # garante que os dicionários existam
        killers = [tag for tag, dropped in getattr(self, "drop_head_mask", {}).items() if dropped]
        reinits = getattr(self, "reinits", {})

        live_strikers = {tag: n for tag, n in self.strikes.items() if n > 0 and tag not in killers}

        print("\n======== HEAD HEALTH REPORT — EPOCH", epoch, "========")

        if killers:
            print("⚔️  HEADS GUILHOTINADAS (zero-out permanente):")
            for tag in sorted(killers):
                n_re = reinits.get(tag, 0)
                print(f"  • {tag:50s} | reinits: {n_re}")
        else:
            print("— Nenhuma cabeça guilhotinada nesta epoch.")

        if live_strikers:
            print("\n⚠️  HEADS COM STRIKES (ainda vivas):")
            for tag, n in sorted(live_strikers.items(), key=lambda x: (-x[1], x[0])):
                print(f"  • {tag:50s} | strikes: {n}")
        else:
            print("\n— Nenhum strike ativo.")

        print("===============================================\n")


    def _debug_tokenization_methods(self, token_ids: torch.Tensor, concept_name: str):
        """Compara os diferentes métodos de tokenização"""
        print(f"\n=== DEBUG TOKENIZATION - {concept_name} ===")
        
        for i, tid in enumerate(token_ids.tolist()):
            # Método atual (problemático)
            convert_result = self.tokenizer_g.convert_ids_to_tokens([tid])[0]
            
            # Método melhor
            decode_result = self.tokenizer_g.decode([tid]).strip()
            
            print(f"  {i:2d}: ID={tid:5d} | convert='{convert_result}' | decode='{decode_result}'")
            
            if convert_result.replace('</w>', '').strip() != decode_result:
                print(f"       ^^^ INCONSISTÊNCIA DETECTADA!")
        
        print("=" * 50)