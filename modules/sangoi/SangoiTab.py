from pathlib import Path

import customtkinter as ctk

from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class SangoiTab:
    """
    A aba para configurações personalizadas do usuário (Sangoi),
    incluindo Delta Pattern e configurações de Camadas.
    """

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        self.refresh_ui()

    def refresh_ui(self):
        """
        Cria e atualiza os elementos da interface do usuário para a aba Sangoi.
        """
        if hasattr(self, 'main_frame') and self.main_frame:
            self.main_frame.destroy()

        self.main_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)

        # Configuração do grid para o frame principal da aba:
        # 6 colunas para acomodar até 3 "slots" de (Label + Widget) por linha.
        # Slot 1: col 0 (Label), col 1 (Widget)
        # Slot 2: col 2 (Label), col 3 (Widget)
        # Slot 3: col 4 (Label), col 5 (Widget)
