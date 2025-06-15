from __future__ import annotations
import datetime as _dt
from typing import Optional

_DEFAULT_FMT = "%Y-%m-%d_%H-%M-%S"
_STYLES = {
    "default":  _DEFAULT_FMT,                 # 2025-06-14_21-54-03
    "compact":  "%Y%m%d-%H%M%S",              # 20250614-215403
    "file":     "%Y%m%d_%H%M%S",              # 20250614_215403
    "readable": "%Y-%m-%d %Hh%Mm%Ss",         # 2025-06-14 21h54m03s
}

def get_string_timestamp(
    *,
    dt: Optional[_dt.datetime] = None,
    tz: Optional[_dt.tzinfo] = None,
    style: str = "default",
    fmt: Optional[str] = None,
) -> str:
    """
    Retorna timestamp em string.

    • style → "default" | "compact" | "file" | "readable"
              (ou qualquer outra chave adicionada ao dicionário _STYLES)
    • fmt   → padrão strftime customizado (tem prioridade sobre style)
    • dt / tz → para testes ou timezone específico.

    Exemplo rápido:
        >>> get_string_timestamp()                  # default
        '2025-06-14_21-54-03'
        >>> get_string_timestamp(style='compact')
        '20250614-215403'
        >>> get_string_timestamp(fmt='%d-%b-%Y_%Hh%M')
        '14-Jun-2025_21h54'
    """
    pattern = fmt or _STYLES.get(style, _DEFAULT_FMT)
    dt_obj = dt or _dt.datetime.now(tz or _dt.datetime.now().astimezone().tzinfo)
    return dt_obj.strftime(pattern)
