import colorama
from colorama import Fore, Style, Back

# Inicializa o colorama (necessário no Windows)
# autoreset=True faz com que cada print() volte à cor padrão automaticamente
colorama.init(autoreset=True)

def logFun(mensagem, lvl="INFO"):
    """
    Imprime uma mensagem de log colorida no console.

    Args:
        mensagem (str): A mensagem a ser exibida.
        lvl (str): O nível do log ('INFO', 'WARNING', 'ERROR', 'DEBUG', 'SUCCESS').
                    Determina a cor da mensagem.
    """
    lvl = lvl.upper() # Garante que o nível seja maiúsculo

    if lvl == "INFO":
        # Azul claro (ciano) para informações gerais
        print(f"{Fore.CYAN}[INFO] {mensagem}")
    elif lvl == "WARNING":
        # Amarelo para avisos
        print(f"{Fore.YELLOW}[WARNING] {mensagem}")
    elif lvl == "ERROR":
        # Vermelho para erros
        print(f"{Fore.RED}[ERROR] {mensagem}")
    elif lvl == "DEBUG":
        # Magenta (ou outra cor) para debug
        print(f"{Fore.MAGENTA}[DEBUG] {mensagem}")
    elif lvl == "SUCCESS":
        # Verde para sucesso
        print(f"{Fore.GREEN}[SUCCESS] {mensagem}")
    else:
        # Cor padrão para níveis desconhecidos
        print(f"[{lvl}] {mensagem}")