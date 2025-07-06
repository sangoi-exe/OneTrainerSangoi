import torch
import tkinter as tk
from tkinter import filedialog

def inspect_pytorch_model_file(file_path: str):
    """
    Carrega e inspeciona um arquivo de modelo PyTorch (.pt, .pth, .bin, .safetensors).
    Mostra a estrutura de alto nível, e para cada tensor no state_dict (se encontrado),
    exibe seu nome, shape e dtype.
    """
    if not file_path:
        print("Nenhum arquivo selecionado.")
        return

    print(f"--- Inspecionando Arquivo: {file_path} ---\n")

    try:
        # Tenta carregar o arquivo. map_location='cpu' para evitar problemas com CUDA
        # se o modelo foi salvo na GPU e você não tem uma, ou não quer usá-la.
        if file_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            # Para safetensors, o carregamento já retorna o state_dict diretamente
            data = load_file(file_path, device="cpu")
            print("Formato: safetensors (carregado diretamente como state_dict)\n")
            # Safetensors já é um state_dict, então processamos diretamente
            if isinstance(data, dict):
                print("Conteúdo é um dicionário (provavelmente state_dict):\n")
                _print_state_dict_info(data)
            else:
                print(f"Conteúdo não é um dicionário. Tipo: {type(data)}")
                print("Tentando imprimir representação do objeto (pode ser grande):")
                print(data)

        else: # Para .pt, .pth, .bin
            data = torch.load(file_path, map_location='cpu', weights_only=False) # Mude para True se tiver certeza que é só state_dict
            print(f"Formato: {file_path.split('.')[-1]} (carregado com torch.load)\n")

            # Verifica se o dado carregado é um state_dict (um dicionário)
            # Modelos completos também são carregados, mas geralmente o state_dict é o que queremos inspecionar
            if isinstance(data, dict):
                # Verifica se parece ser um state_dict (contém tensores)
                # ou se é um dicionário de metadados de checkpoints mais complexos (como os do HF)
                is_likely_state_dict = any(isinstance(v, torch.Tensor) for v in data.values())

                if is_likely_state_dict:
                    print("Conteúdo é um dicionário (provavelmente state_dict):\n")
                    _print_state_dict_info(data)
                else:
                    print("Conteúdo é um dicionário, mas pode não ser um state_dict simples (ex: checkpoint HF com config).")
                    print("Chaves de alto nível:")
                    for key in data.keys():
                        print(f"- {key} (Tipo: {type(data[key])})")
                    # Se houver uma chave comum para state_dict, como 'model' ou 'state_dict', inspeciona ela
                    common_sd_keys = ['model', 'state_dict', 'model_state_dict']
                    found_sd_key = None
                    for sd_key in common_sd_keys:
                        if sd_key in data and isinstance(data[sd_key], dict):
                            found_sd_key = sd_key
                            break
                    if found_sd_key:
                        print(f"\n--- Inspecionando state_dict aninhado sob a chave: '{found_sd_key}' ---\n")
                        _print_state_dict_info(data[found_sd_key])
                    else:
                        print("\nNão foi encontrado um state_dict aninhado óbvio. Imprimindo chaves e tipos do dicionário principal:")
                        for key, value in data.items():
                            if isinstance(value, torch.Tensor):
                                print(f"  Chave: {key}, Shape: {value.shape}, Dtype: {value.dtype}")
                            else:
                                print(f"  Chave: {key}, Tipo: {type(value)}")


            # Se não for um dicionário, pode ser um modelo inteiro salvo
            elif hasattr(data, 'state_dict') and callable(data.state_dict):
                print("Conteúdo parece ser um objeto de modelo PyTorch completo.")
                print(f"Tipo do Modelo: {type(data)}\n")
                print("--- Inspecionando state_dict do modelo ---\n")
                _print_state_dict_info(data.state_dict())
                print("\n--- Estrutura do Modelo (representação string) ---")
                print(data) # Imprime a estrutura do modelo
            else:
                print(f"Conteúdo carregado não é um dicionário nem um objeto de modelo PyTorch reconhecível.")
                print(f"Tipo: {type(data)}")
                print("Tentando imprimir representação do objeto (pode ser grande):")
                print(data)

    except Exception as e:
        print(f"Erro ao carregar ou inspecionar o arquivo: {e}")
        import traceback
        traceback.print_exc()

def _print_state_dict_info(state_dict: dict):
    """Função auxiliar para imprimir informações de um state_dict."""
    total_params = 0
    param_count_per_layer = {}

    print(f"{'Nome da Camada/Peso':<80} {'Shape':<25} {'Dtype':<15} {'Nº Parâmetros':<15}")
    print("-" * 140)

    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params
            param_count_per_layer[name] = num_params
            print(f"{name:<80} {str(param.shape):<25} {str(param.dtype):<15} {num_params:<15,}")
        else:
            print(f"{name:<80} {'N/A (Não é Tensor)':<25} {str(type(param)):<15} {'N/A':<15}")

    print("-" * 140)
    print(f"\nTotal de parâmetros no state_dict: {total_params:,}")

    # Opcional: Mostrar as camadas com mais parâmetros
    # sorted_layers = sorted(param_count_per_layer.items(), key=lambda item: item[1], reverse=True)
    # print("\nTop 5 camadas por número de parâmetros:")
    # for i, (name, count) in enumerate(sorted_layers[:5]):
    #     print(f"{i+1}. {name}: {count:,}")

def main():
    root = tk.Tk()
    root.withdraw() # Esconde a janela principal do Tkinter
    root.call('wm', 'attributes', '.', '-topmost', True) # Mantém a filedialog no topo

    file_path = filedialog.askopenfilename(
        title="Selecione o arquivo PyTorch (.pt, .pth, .bin, .safetensors)",
        filetypes=(
            ("Arquivos PyTorch", "*.pt *.pth *.bin *.safetensors"),
            ("Todos os arquivos", "*.*")
        )
    )

    if file_path:
        inspect_pytorch_model_file(file_path)
    else:
        print("Nenhuma arquivo selecionado. Encerrando.")

if __name__ == "__main__":
    main()