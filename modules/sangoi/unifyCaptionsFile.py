import os
import tkinter as tk
from tkinter import filedialog

def combinar_legendas():
    """
    Script para combinar legendas de 4 fontes diferentes em um único arquivo de texto por imagem.
    Cada legenda de origem ocupará uma nova linha no arquivo de destino.
    """
    
    # --- Passo 1: Configurar a janela raiz do Tkinter para as caixas de diálogo ---
    # Isso é necessário para que as caixas de diálogo funcionem corretamente.
    # A janela principal fica oculta, pois só precisamos das caixas de diálogo.
    root = tk.Tk()
    root.withdraw()  # Oculta a janela principal do Tkinter
    
    # Define a propriedade "topmost" para que as caixas de diálogo fiquem sempre em primeiro plano
    root.attributes('-topmost', True)

    print("--- Script para Combinar Legendas ---")
    print("Você precisará selecionar as 4 pastas de entrada e 1 pasta de saída.\n")

    # --- Passo 2: Solicitar as 4 pastas de entrada ---
    pastas_de_entrada = []
    for i in range(4):
        titulo_dialogo = f"Por favor, selecione a pasta de legendas FONTE #{i + 1}"
        pasta = filedialog.askdirectory(title=titulo_dialogo)
        
        # Se o usuário cancelar a seleção, encerra o script
        if not pasta:
            print("\nOperação cancelada pelo usuário. Encerrando.")
            return
        
        pastas_de_entrada.append(pasta)
        print(f"Pasta Fonte #{i + 1} selecionada: {pasta}")

    # --- Passo 3: Solicitar a pasta de saída ---
    pasta_de_saida = filedialog.askdirectory(title="Selecione a pasta de SAÍDA para salvar os arquivos combinados")
    
    if not pasta_de_saida:
        print("\nOperação cancelada pelo usuário. Encerrando.")
        return
    
    print(f"Pasta de Saída selecionada: {pasta_de_saida}\n")
    
    # Cria a pasta de saída se ela não existir
    os.makedirs(pasta_de_saida, exist_ok=True)

    # --- Passo 4: Processar os arquivos ---
    # Usaremos a primeira pasta como referência para obter a lista de nomes de arquivos.
    # O script assume que os nomes dos arquivos são os mesmos em todas as pastas.
    pasta_referencia = pastas_de_entrada[0]
    
    print(f"Processando arquivos da pasta de referência: {pasta_referencia}")
    
    arquivos_processados = 0
    arquivos_ignorados = 0

    # Itera sobre cada arquivo na pasta de referência
    for nome_arquivo in os.listdir(pasta_referencia):
        # Processa apenas arquivos .txt para evitar outros tipos de arquivo
        if nome_arquivo.endswith(".txt"):
            legendas_combinadas = []
            
            # Itera sobre cada uma das 4 pastas de entrada
            for pasta in pastas_de_entrada:
                caminho_arquivo_fonte = os.path.join(pasta, nome_arquivo)
                
                try:
                    with open(caminho_arquivo_fonte, 'r', encoding='utf-8') as f:
                        # Lê a linha única, remove espaços em branco e quebras de linha extras
                        legenda = f.readline().strip()
                        if legenda:  # Adiciona apenas se a linha não estiver vazia
                            legendas_combinadas.append(legenda)
                except FileNotFoundError:
                    print(f"AVISO: O arquivo '{nome_arquivo}' não foi encontrado em '{pasta}'. Será ignorado para esta fonte.")
                except Exception as e:
                    print(f"ERRO: Não foi possível ler o arquivo '{caminho_arquivo_fonte}'. Erro: {e}")

            # Se alguma legenda foi coletada, salva no novo arquivo
            if legendas_combinadas:
                caminho_arquivo_saida = os.path.join(pasta_de_saida, nome_arquivo)
                
                with open(caminho_arquivo_saida, 'w', encoding='utf-8') as f_out:
                    # Junta as legendas com um caractere de nova linha
                    f_out.write('\n'.join(legendas_combinadas))
                
                arquivos_processados += 1
        else:
            arquivos_ignorados += 1

    # --- Passo 5: Exibir o resumo ---
    print("\n--- Processo Concluído! ---")
    print(f"Total de arquivos de legenda combinados: {arquivos_processados}")
    if arquivos_ignorados > 0:
        print(f"Total de arquivos ignorados (não .txt): {arquivos_ignorados}")
    print(f"Os novos arquivos foram salvos em: {pasta_de_saida}")


# Executa a função principal quando o script é iniciado
if __name__ == "__main__":
    combinar_legendas()