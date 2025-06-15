import logging
import threading
from rich.console import Console as RichConsole
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
from typing import Optional

# Console global padrão
_default_logfun_console = RichConsole()
_current_logfun_console = _default_logfun_console

# Progress manager global
_global_progress: Optional[Progress] = None

# _progress_live: Optional[Live] = None 
_progress_lock = threading.Lock()

def set_logfun_console(console: RichConsole) -> None:
    """Define o console que o logFun vai usar por padrão."""
    global _current_logfun_console
    _current_logfun_console = console

def init_global_progress() -> Progress:
    """
    Inicializa o sistema de progresso global (apenas o objeto Progress, sem iniciar Live).
    O objeto Progress retornado deve ser adicionado a um Rich Live display externo.
    """
    global _global_progress
    
    with _progress_lock:
        if _global_progress is None:
            _global_progress = Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                TextColumn("• {task.speed:.2f} it/s"),
                console=_current_logfun_console,
                transient=False, # Tasks added to this progress object will remain after completion
                refresh_per_second=4
            )
    
    return _global_progress

def cleanup_global_progress():
    """Limpa o sistema de progresso global. (Apenas reseta _global_progress)."""
    global _global_progress
    
    with _progress_lock:
        if _global_progress:
            _global_progress = None

def create_progress_task(description: str, total: int) -> int:
    """
    Cria uma nova task de progresso.
    
    Args:
        description: Descrição da task
        total: Total de itens
        
    Returns:
        task_id para usar nas atualizações
    """
    progress = init_global_progress() 
    return progress.add_task(description, total=total)

def update_progress_task(task_id: int, advance: int = 1, **kwargs):
    """Atualiza uma task de progresso."""
    if _global_progress:
        _global_progress.update(task_id, advance=advance, **kwargs)

def remove_progress_task(task_id: int):
    """Remove uma task de progresso."""
    if _global_progress:
        _global_progress.remove_task(task_id)

def logFun(mensagem: str, lvl: str = "INFO", _console: Optional[RichConsole] = None) -> None:
    """
    Imprime uma mensagem de log colorida no console.
    
    Args:
        mensagem: A mensagem a ser exibida
        lvl: O nível do log
        _console: Console específico (opcional)
    """
    console_to_use = _console or _current_logfun_console
    level_upper = lvl.upper()

    # NOTA: O logFun agora pode imprimir sobre o Live display sem problemas,
    # desde que seja no console que o Live está usando.
    # O Live automaticamente suspende sua exibição para permitir o print,
    # e depois retoma.

    match level_upper:
        case "INFO":
            console_to_use.print(f"[dark_olive_green1][INFO][/dark_olive_green1] [sky_blue1]{mensagem}[/sky_blue1]")
        case "LOOP":
            console_to_use.print(f"[cyan][TRAINER][/cyan] [sky_blue1]{mensagem}[/sky_blue1]")
        case "CONVCTRL":
            console_to_use.print(f"[light_salmon1][CONVCTRL][/light_salmon1] [sky_blue1]{mensagem}[/sky_blue1]")
        case "TRAINGPS":
            console_to_use.print(f"[light_steel_blue3][TRAINGPS][/light_steel_blue3] [sky_blue1]{mensagem}[/sky_blue1]")
        case "LORA":
            console_to_use.print(f"[slate_blue1][LORA][/slate_blue1] [sky_blue1]{mensagem}[/sky_blue1]")
        case "VERBOSE":
            console_to_use.print(f"[orange3][VERBOSE][/orange3] [sky_blue1]{mensagem}[/sky_blue1]")
        case "WARNING":
            console_to_use.print(f"[gold1][WARNING][/gold1] [sky_blue1]{mensagem}[/sky_blue1]")
        case "ERROR":
            console_to_use.print(f"[red][ERROR][/red] [pink1]{mensagem}[/pink1]", soft_wrap=True)
        case "DEBUG":
            console_to_use.print(f"[grey35][DEBUG][/grey35] [sky_blue1]{mensagem}[/sky_blue1]")
        case "SUCCESS":
            console_to_use.print(f"[spring_green3][SUCCESS][/spring_green3] [sky_blue1]{mensagem}[/sky_blue1]")
        case "PROGRESS":
            console_to_use.print(f"[medium_purple1][PROGRESS][/medium_purple1] [sky_blue1]{mensagem}[/sky_blue1]")
        case _:
            console_to_use.print(f"[white][{level_upper}][/white] [sky_blue1]{mensagem}[/sky_blue1]")

# Context manager para facilitar o uso (mantém-se o mesmo, pois depende de _global_progress)
class ProgressContext:
    """Context manager para barras de progresso."""
    
    def __init__(self, description: str, total: int):
        self.description = description
        self.total = total
        self.task_id = None
    
    def __enter__(self):
        self.task_id = create_progress_task(self.description, self.total)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.task_id is not None:
            remove_progress_task(self.task_id)
    
    def update(self, advance: int = 1, **kwargs):
        """Atualiza o progresso."""
        if self.task_id is not None:
            update_progress_task(self.task_id, advance=advance, **kwargs)