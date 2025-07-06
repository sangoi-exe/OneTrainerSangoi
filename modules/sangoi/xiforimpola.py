import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
import tkinter as tk
from tkinter import filedialog, messagebox

# Prefixes para as chaves do text encoder do SDXL
SDXL_TEXT_PREFIXES = [
    "conditioner.embedders.1.model.", # Prefixo padrão
    "embedders.1.model."              # Alternativa se "conditioner." estiver faltando
]

def strip_sdxl_prefix(key: str) -> str:
    """Remove prefixos conhecidos do text encoder do SDXL."""
    for p in SDXL_TEXT_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    # Se chegou aqui, o prefixo não foi encontrado como esperado.
    # Isso pode indicar um problema com o arquivo SDXL ou os prefixos definidos.
    print(f"AVISO: Prefixo esperado não encontrado na chave SDXL: '{key}'. Verifique os SDXL_TEXT_PREFIXES.")
    return key

def choose_file(title, pattern):
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title=title, filetypes=[pattern])
    root.destroy()
    return Path(path) if path else None

def choose_save(title, ext):
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.asksaveasfilename(title=title, defaultextension=ext,
                                        filetypes=[("PyTorch bin", "*.bin")])
    root.destroy()
    return Path(path) if path else None

def merge(sdxl_path: Path, bigg_path: Path, dst_path: Path):
    print(f"Carregando modelo SDXL (text encoder) de: {sdxl_path}")
    sdxl_state_dict = load_file(str(sdxl_path))
    print(f"Carregando modelo ViT-BigG original de: {bigg_path}")
    bigg_original_state_dict = torch.load(str(bigg_path), map_location="cpu")

    # Criamos uma cópia do state_dict do BigG original.
    # Todas as camadas do BigG (incluindo as visuais) serão mantidas,
    # a menos que sejam explicitamente substituídas pelas camadas de texto do SDXL.
    merged_state_dict = bigg_original_state_dict.copy()

    replaced_count = 0
    skipped_sdxl_layers_details = [] # Para armazenar detalhes das camadas ignoradas

    print("\nIniciando a mesclagem de camadas (SDXL Text Encoder -> BigG Text Encoder):")
    for sdxl_key, sdxl_tensor in sdxl_state_dict.items():
        # Remove o prefixo do SDXL para obter o nome base da camada
        # Ex: "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias"
        # torna-se "transformer.resblocks.0.ln_1.bias"
        base_key_from_sdxl = strip_sdxl_prefix(sdxl_key)

        # A chave de destino no BigG para as camadas de TEXTO é o base_key_from_sdxl
        # NÃO adicionamos "visual." aqui.
        bigg_target_key = base_key_from_sdxl

        if bigg_target_key in merged_state_dict:
            bigg_original_tensor_at_target = merged_state_dict[bigg_target_key]
            
            # Verificamos se a camada no BigG é realmente uma camada de texto (não tem 'visual.')
            # Esta verificação é uma segurança extra, mas o mapeamento direto já deve cuidar disso.
            if bigg_target_key.startswith("visual."):
                print(f"  ALERTA DE MAPEAMENTO INCORRETO: Chave SDXL '{sdxl_key}' (base '{base_key_from_sdxl}') "
                      f"está tentando mapear para uma chave VISUAL '{bigg_target_key}' no BigG. Isso não deveria acontecer. Ignorando.")
                skipped_sdxl_layers_details.append(f"{sdxl_key} -> {bigg_target_key} (tentativa de sobrescrever visual, ignorado)")
                continue

            if bigg_original_tensor_at_target.shape == sdxl_tensor.shape:
                original_bigg_dtype = bigg_original_tensor_at_target.dtype
                
                if sdxl_tensor.dtype != original_bigg_dtype:
                    print(f"  Substituindo e convertendo tipo da camada DE TEXTO '{bigg_target_key}': "
                          f"{sdxl_tensor.dtype} (SDXL) -> {original_bigg_dtype} (BigG)")
                    merged_state_dict[bigg_target_key] = sdxl_tensor.to(original_bigg_dtype)
                else:
                    print(f"  Substituindo camada DE TEXTO '{bigg_target_key}': mantendo {original_bigg_dtype} (mesmo tipo)")
                    merged_state_dict[bigg_target_key] = sdxl_tensor
                replaced_count += 1
            else:
                msg = (f"  IGNORANDO (shape incompatível): Chave SDXL '{sdxl_key}' (base '{base_key_from_sdxl}') "
                       f"mapeia para '{bigg_target_key}' (texto) no BigG, mas os shapes divergem. "
                       f"SDXL: {sdxl_tensor.shape}, BigG: {bigg_original_tensor_at_target.shape}")
                print(msg)
                skipped_sdxl_layers_details.append(msg)
        else:
            msg = (f"  IGNORANDO (não encontrada no BigG): Chave SDXL '{sdxl_key}' (base '{base_key_from_sdxl}') "
                   f"mapeia para '{bigg_target_key}', que não existe nas camadas de texto do modelo BigG.")
            print(msg)
            skipped_sdxl_layers_details.append(msg)

    print(f"\nSalvando modelo mesclado em: {dst_path}")
    torch.save(merged_state_dict, str(dst_path))
    
    info_message = (
        f"Camadas do Text Encoder do SDXL substituídas no Text Encoder do BigG: {replaced_count}\n"
        f"Camadas do Text Encoder do SDXL ignoradas: {len(skipped_sdxl_layers_details)}\n"
        f"Modelo salvo em (precisão original do BigG preservada para todas as camadas, incluindo visuais):\n{dst_path}"
    )
    if skipped_sdxl_layers_details:
        info_message += "\n\nDetalhes das primeiras camadas ignoradas:\n"
        for i, detail in enumerate(skipped_sdxl_layers_details):
            if i < 10: # Mostrar até 10 detalhes para não poluir a messagebox
                info_message += f"\n  - {detail.split(': ', 1)[1] if ': ' in detail else detail}" # Simplifica a mensagem
            else:
                info_message += f"\n  ... e mais {len(skipped_sdxl_layers_details) - 10} camadas."
                break
                
    messagebox.showinfo("Concluído", info_message)

if __name__ == "__main__":
    sdxl = choose_file("Selecione o SDXL Text Encoder (.safetensors)", ("Safetensors", "*.safetensors"))
    if not sdxl: sys.exit("Cancelado.")
    bigg = choose_file("Selecione o ViT-Big G original (.bin)", ("PyTorch bin", "*.bin"))
    if not bigg: sys.exit("Cancelado.")
    out  = choose_save("Salvar novo modelo Big-G com Text Encoder do SDXL", ".bin")
    if not out: sys.exit("Cancelado.")
    
    try:
        merge(sdxl, bigg, out)
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        messagebox.showerror("Erro", f"Ocorreu um erro durante o processamento:\n{e}")
        sys.exit(f"Erro: {e}")