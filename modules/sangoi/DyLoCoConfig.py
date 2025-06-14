from typing import Any
from modules.util.config.BaseConfig import BaseConfig

class DyLoCoConfig(BaseConfig):
    loss_type: str
    start: float
    end: float

    def __init__(self, data: list[tuple[str, Any, type, bool]]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []
        data.append(("loss_type", "", str, False))
        data.append(("start", 0.0, float, False))
        data.append(("end",  0.0, float, False))
        return DyLoCoConfig(data)
