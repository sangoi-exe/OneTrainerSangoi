import json
import threading
import traceback
import webbrowser
from collections.abc import Callable
from pathlib import Path
from tkinter import PhotoImage, filedialog

from modules.ui.AdditionalEmbeddingsTab import AdditionalEmbeddingsTab

from modules.ui.CloudTab import CloudTab
from modules.ui.ConceptTab import ConceptTab
from modules.ui.ConvertModelUI import ConvertModelUI
from modules.ui.LoraTab import LoraTab
from modules.ui.ModelTab import ModelTab
from modules.ui.ProfilingWindow import ProfilingWindow
from modules.ui.SampleWindow import SampleWindow
from modules.ui.SamplingTab import SamplingTab
from modules.ui.TopBar import TopBar
from modules.ui.TrainingTab import TrainingTab
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.zluda import ZLUDA

import torch

import customtkinter as ctk
from customtkinter import AppearanceModeTracker


class TrainUI(ctk.CTk):
    set_step_progress: Callable[[int, int], None]
    set_epoch_progress: Callable[[int, int], None]

    status_label: ctk.CTkLabel | None
    training_button: ctk.CTkButton | None
    training_callbacks: TrainCallbacks | None
    training_commands: TrainCommands | None

    general_tab: ctk.CTkFrame | None = None
    model_tab: ModelTab | None = None
    data_tab: ctk.CTkFrame | None = None
    concepts_tab: ConceptTab | None = None
    training_tab: TrainingTab | None = None
    sampling_tab_content: ctk.CTkFrame | None = None  # Renomeado para evitar conflito com o método
    backup_tab: ctk.CTkFrame | None = None
    tools_tab: ctk.CTkFrame | None = None
    additional_embeddings_tab: AdditionalEmbeddingsTab | None = None
    cloud_tab: CloudTab | None = None
    lora_tab_content: LoraTab | None = None  # Renomeado para evitar conflito com o método
    embedding_tab_content: ctk.CTkFrame | None = None  # Renomeado para evitar conflito com o método
    tabview: ctk.CTkTabview | None = None
    pause_switch_widget: ctk.CTkSwitch | None = None
    pause_switch_var: ctk.BooleanVar | None = None  # Nova variável

    def __init__(self):
        super().__init__()

        self.title("OneTrainer")
        try:
            self.iconbitmap("resources/icons/icon.ico")
        except Exception as e:
            print(f"Warn: Could not load iconbitmap: {e}")  # Melhor log

        try:
            self._icon_photo = PhotoImage(file="resources/icons/icon.png")
            self.wm_iconphoto(True, self._icon_photo)
        except Exception as e:
            print(f"Warn: Could not load icon photo: {e}")  # Melhor log

        self.geometry("1100x740")

        ctk.set_appearance_mode("Light" if AppearanceModeTracker.detect_appearance_mode() == 0 else "Dark")
        ctk.set_default_color_theme("blue")

        self.train_config = TrainConfig.default_values()
        self.ui_state = UIState(self, self.train_config)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.status_label = None
        self.training_button = None
        self.export_button = None
        # self.tabview inicializado em content_frame

        # Inicialização das barras superior e inferior (sem alterações significativas)
        self.top_bar_component = self.top_bar(self)
        self.content_frame(self)  # Cria a estrutura das abas e carrega a primeira
        self.bottom_bar(self)

        self.training_thread = None
        self.training_callbacks = None
        self.training_commands = None

        # Persistent profiling window.
        self.profiling_window = ProfilingWindow(self)

        self.protocol("WM_DELETE_WINDOW", self.__close)

        # START: Otimização - Chamar change_training_method após UI básica estar pronta
        # Garante que as abas condicionais (LoRA/Embedding) sejam adicionadas/removidas corretamente
        # na estrutura inicial, antes do lazy loading do conteúdo.
        self.change_training_method(self.train_config.training_method)
        # END: Otimização - Chamar change_training_method após UI básica estar pronta
        self.pause_switch_var = ctk.BooleanVar(self, value=False)

    # __close, top_bar, bottom_bar permanecem iguais
    def __close(self):
        self.top_bar_component.save_default()
        print("[UI] Janela fechada. O treinamento continua em segundo plano.")
        self.withdraw()

    def top_bar(self, master):
        # Sem alterações necessárias aqui para a otimização
        return TopBar(
            master,
            self.train_config,
            self.ui_state,
            self.change_model_type,
            self.change_training_method,
            self.load_preset,
        )

    def bottom_bar(self, master):
        # Sem alterações necessárias aqui para a otimização
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=2, column=0, sticky="nsew")

        self.set_step_progress, self.set_epoch_progress = components.double_progress(frame, 0, 0, "step", "epoch")

        self.status_label = components.label(frame, 0, 1, "", tooltip="Current status of the training run")

        frame.grid_columnconfigure(2, weight=1)  # padding

        components.button(frame, 0, 3, "Tensorboard", self.open_tensorboard)
        self.training_button = components.button(frame, 0, 4, "Start Training", self.start_training)
        self.export_button = components.button(
            frame,
            0,
            5,
            "Export",
            self.export_training,
            tooltip="Export the current configuration as a script to run without a UI",
        )

        return frame

    def content_frame(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(frame, command=self._on_tab_change)  # Adiciona comando para lazy loading
        self.tabview.grid(row=0, column=0, sticky="nsew")

        # START: Otimização - Adicionar apenas os nomes das abas
        self.tabview.add("general")
        self.tabview.add("model")
        self.tabview.add("data")
        self.tabview.add("concepts")
        self.tabview.add("training")
        self.tabview.add("sampling")
        self.tabview.add("backup")
        self.tabview.add("tools")
        self.tabview.add("additional embeddings")
        self.tabview.add("cloud")
        # Abas condicionais (LoRA/Embedding) são adicionadas/removidas em change_training_method

        # Carrega o conteúdo da primeira aba visível ("general") imediatamente
        self._on_tab_change()
        # END: Otimização - Adicionar apenas os nomes das abas

        # self.change_training_method(self.train_config.training_method) # Movido para o final de __init__

        return frame

    # START: Otimização - Handler para carregar conteúdo da aba sob demanda
    def _on_tab_change(self):
        """Callback executado quando o usuário troca de aba. Carrega o conteúdo da aba selecionada se ainda não foi carregado."""
        if not self.tabview:
            return  # Segurança

        selected_tab_name = self.tabview.get()
        if selected_tab_name is None:
            return  # Pode acontecer durante a inicialização/destruição

        tab_widget = self.tabview.tab(selected_tab_name)
        if not tab_widget:
            print(f"Warn: Tab widget for '{selected_tab_name}' not found during lazy load.")
            return

        # Usa um mapeamento para simplificar a lógica
        tab_creation_map = {
            "general": ("general_tab", self.create_general_tab),
            "model": ("model_tab", self.create_model_tab),
            "data": ("data_tab", self.create_data_tab),
            "concepts": ("concepts_tab", self.create_concepts_tab),
            "training": ("training_tab", self.create_training_tab),
            "sampling": ("sampling_tab_content", self.create_sampling_tab),
            "backup": ("backup_tab", self.create_backup_tab),
            "tools": ("tools_tab", self.create_tools_tab),
            "additional embeddings": ("additional_embeddings_tab", self.create_additional_embeddings_tab),
            "cloud": ("cloud_tab", self.create_cloud_tab),
            "LoRA": ("lora_tab_content", self.lora_tab),
            "embedding": ("embedding_tab_content", self.embedding_tab),
        }

        if selected_tab_name in tab_creation_map:
            attr_name, creation_func = tab_creation_map[selected_tab_name]
            # Verifica se o atributo correspondente (e.g., self.general_tab) é None
            if getattr(self, attr_name, None) is None:
                print(f"[UI] Lazy loading tab: {selected_tab_name}")
                # Cria o conteúdo da aba e armazena na variável de instância
                created_content = creation_func(tab_widget)
                setattr(self, attr_name, created_content)

    # Métodos create_*_tab permanecem quase idênticos,
    # apenas garantindo que usem o 'master' recebido corretamente.
    # O retorno do frame/widget criado é importante para o lazy loading.

    def create_general_tab(self, master):
        # Conteúdo original do método, garantindo usar 'master'
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        components.label(
            frame, 0, 0, "Workspace Directory", tooltip="The directory where all files of this training run are saved"
        )
        components.dir_entry(frame, 0, 1, self.ui_state, "workspace_dir")
        components.label(frame, 1, 0, "Cache Directory", tooltip="The directory where cached data is saved")
        components.dir_entry(frame, 1, 1, self.ui_state, "cache_dir")
        components.label(frame, 2, 0, "Continue from last backup", tooltip="...")
        components.switch(frame, 2, 1, self.ui_state, "continue_last_backup")
        components.label(frame, 3, 0, "Only Cache", tooltip="...")
        components.switch(frame, 3, 1, self.ui_state, "only_cache")
        components.label(frame, 4, 0, "Debug mode", tooltip="...")
        components.switch(frame, 4, 1, self.ui_state, "debug_mode")
        components.label(frame, 5, 0, "Debug Directory", tooltip="...")
        components.dir_entry(frame, 5, 1, self.ui_state, "debug_dir")
        components.label(frame, 6, 0, "Tensorboard", tooltip="...")
        components.switch(frame, 6, 1, self.ui_state, "tensorboard")
        components.label(frame, 7, 0, "Expose Tensorboard", tooltip="...")
        components.switch(frame, 7, 1, self.ui_state, "tensorboard_expose")
        components.label(frame, 7, 2, "Tensorboard Port", tooltip="...")
        components.entry(frame, 7, 3, self.ui_state, "tensorboard_port")
        components.label(frame, 8, 0, "Validation", tooltip="...")
        components.switch(frame, 8, 1, self.ui_state, "validation")
        components.label(frame, 9, 0, "Validate after", tooltip="...")
        components.time_entry(frame, 9, 1, self.ui_state, "validate_after", "validate_after_unit")
        components.label(frame, 10, 0, "Dataloader Threads", tooltip="...")
        components.entry(frame, 10, 1, self.ui_state, "dataloader_threads")
        components.label(frame, 11, 0, "Train Device", tooltip="...")
        components.entry(frame, 11, 1, self.ui_state, "train_device")
        components.label(frame, 12, 0, "Temp Device", tooltip="...")
        components.entry(frame, 12, 1, self.ui_state, "temp_device")

        # Importante: Usar pack/grid dentro do frame criado, relativo ao 'master'
        frame.pack(fill="both", expand=True)
        return frame  # Retorna o frame criado

    def create_model_tab(self, master) -> ModelTab:
        # A classe ModelTab já deve lidar com seu próprio 'master'
        model_tab_instance = ModelTab(master, self.train_config, self.ui_state)
        # ModelTab provavelmente chama pack/grid internamente, não precisamos fazer aqui
        return model_tab_instance  # Retorna a instância criada

    def create_data_tab(self, master):
        # Conteúdo original do método, garantindo usar 'master'
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        components.label(frame, 0, 0, "Aspect Ratio Bucketing", tooltip="...")
        components.switch(frame, 0, 1, self.ui_state, "aspect_ratio_bucketing")
        components.label(frame, 1, 0, "Latent Caching", tooltip="...")
        components.switch(frame, 1, 1, self.ui_state, "latent_caching")
        components.label(frame, 2, 0, "Clear cache before training", tooltip="...")
        components.switch(frame, 2, 1, self.ui_state, "clear_cache_before_training")

        frame.pack(fill="both", expand=True)
        return frame  # Retorna o frame criado

    def create_concepts_tab(self, master) -> ConceptTab:
        # A classe ConceptTab já deve lidar com seu próprio 'master'
        concepts_tab_instance = ConceptTab(master, self.train_config, self.ui_state)
        # ConceptTab provavelmente chama pack/grid internamente
        return concepts_tab_instance  # Retorna a instância criada

    def create_training_tab(self, master) -> TrainingTab:
        # A classe TrainingTab já deve lidar com seu próprio 'master'
        training_tab_instance = TrainingTab(master, self.train_config, self.ui_state)
        # TrainingTab provavelmente chama pack/grid internamente
        return training_tab_instance  # Retorna a instância criada

    def create_cloud_tab(self, master) -> CloudTab:
        # A classe CloudTab já deve lidar com seu próprio 'master'
        cloud_tab_instance = CloudTab(master, self.train_config, self.ui_state, parent=self)
        # CloudTab provavelmente chama pack/grid internamente
        return cloud_tab_instance  # Retorna a instância criada

    def create_sampling_tab(self, master):
        # --- Início do conteúdo original de create_sampling_tab ---
        # O 'master' aqui é o widget da aba específica (e.g., self.tabview.tab("sampling"))
        # Precisamos garantir que tudo seja colocado dentro deste 'master'.

        # Frame principal para esta aba
        tab_frame = ctk.CTkFrame(master=master, fg_color="transparent")
        tab_frame.pack(fill="both", expand=True)

        tab_frame.grid_rowconfigure(0, weight=0)
        tab_frame.grid_rowconfigure(1, weight=1)
        tab_frame.grid_columnconfigure(0, weight=1)

        # sample after (Top Frame)
        top_frame = ctk.CTkFrame(master=tab_frame, corner_radius=0)  # Use tab_frame como master
        top_frame.grid(row=0, column=0, sticky="nsew")
        # --- Configuração do top_frame ---
        top_frame.grid_columnconfigure(8, weight=1)  # Adiciona weight para empurrar botões para esquerda

        components.label(top_frame, 0, 0, "Sample After", tooltip="...")
        components.time_entry(top_frame, 0, 1, self.ui_state, "sample_after", "sample_after_unit")
        components.label(top_frame, 0, 2, "Skip First", tooltip="...")
        components.entry(top_frame, 0, 3, self.ui_state, "sample_skip_first", width=50, sticky="nw")
        components.label(top_frame, 0, 4, "Format", tooltip="...")
        components.options_kv(
            top_frame, 0, 5, [("PNG", ImageFormat.PNG), ("JPG", ImageFormat.JPG)], self.ui_state, "sample_image_format"
        )
        components.button(top_frame, 0, 6, "sample now", self.sample_now)
        components.button(top_frame, 0, 7, "manual sample", self.open_sample_ui)

        # Sub Frame dentro do top_frame
        sub_frame = ctk.CTkFrame(master=top_frame, corner_radius=0, fg_color="transparent")
        # Coloca sub_frame abaixo dos controles principais no top_frame
        sub_frame.grid(row=1, column=0, sticky="nsew", columnspan=8)  # Span all columns used above
        # --- Configuração do sub_frame ---
        components.label(sub_frame, 0, 0, "Non-EMA Sampling", tooltip="...")
        components.switch(sub_frame, 0, 1, self.ui_state, "non_ema_sampling")
        components.label(sub_frame, 0, 2, "Samples to Tensorboard", tooltip="...")
        components.switch(sub_frame, 0, 3, self.ui_state, "samples_to_tensorboard")
        components.label(sub_frame, 0, 4, "Skip samples on train start", tooltip="...")
        components.switch(sub_frame, 0, 5, self.ui_state, "skip_sample_on_train_start")

        # Tabela (SamplingTab)
        # Frame para conter a tabela, abaixo do top_frame
        table_container_frame = ctk.CTkFrame(master=tab_frame, corner_radius=0)  # Use tab_frame como master
        table_container_frame.grid(row=1, column=0, sticky="nsew")

        # Instancia SamplingTab dentro do container apropriado
        SamplingTab(table_container_frame, self.train_config, self.ui_state)
        # --- Fim do conteúdo original de create_sampling_tab ---

        return tab_frame  # Retorna o frame principal da aba

    def create_backup_tab(self, master):
        # Conteúdo original do método, garantindo usar 'master'
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)
        row_index = 0

        components.label(frame, 0, 0, "Backup After", tooltip="...")
        components.time_entry(frame, 0, 1, self.ui_state, "backup_after", "backup_after_unit")
        components.button(frame, 0, 3, "backup now", self.backup_now)
        components.label(frame, 1, 0, "Rolling Backup", tooltip="...")
        components.switch(frame, 1, 1, self.ui_state, "rolling_backup")
        components.label(frame, 1, 3, "Rolling Backup Count", tooltip="...")
        components.entry(frame, 1, 4, self.ui_state, "rolling_backup_count")
        components.label(frame, 2, 0, "Backup Before Save", tooltip="...")
        components.switch(frame, 2, 1, self.ui_state, "backup_before_save")
        components.label(frame, 3, 0, "Save Every", tooltip="...")
        components.time_entry(frame, 3, 1, self.ui_state, "save_every", "save_every_unit")
        components.button(frame, 3, 3, "save now", self.save_now)
        components.label(frame, 4, 0, "Skip First", tooltip="...")
        components.entry(frame, 4, 1, self.ui_state, "save_skip_first", width=50, sticky="nw")
        components.label(frame, 5, 0, "Save Filename Prefix", tooltip="...")
        components.entry(frame, 5, 1, self.ui_state, "save_filename_prefix")

        components.label(
            frame,
            6, # Deve ser 6
            0,
            "Pause Training",
            tooltip="Pause after the current epoch finishes and move model to CPU. Toggle again to resume.",
        )

        # Switch na coluna 1 da mesma linha (6)
        self.pause_switch_widget = ctk.CTkSwitch(
            master=frame,
            text="",  # Texto vazio, pois a label está separada
            variable=self.pause_switch_var,  # Usa a variável dedicada
            onvalue=True,
            offvalue=False,
            command=self.toggle_pause,
        )
        self.pause_switch_widget.grid(
            row=6, # Deve ser 6
            column=1,      # Coluna correta ao lado da label
            padx=(0, 20),
            pady=5,
            sticky="w"     # Alinha à esquerda na célula
        )

        frame.pack(fill="both", expand=True)
        return frame  # Retorna o frame criado

    def _update_pause_switch_initial_state(self):
        """Atualiza o estado inicial/atual do switch de pausa se ele existir."""
        if not self.pause_switch_widget:
            return

        is_training = self.training_thread is not None and self.training_thread.is_alive()
        trainer_is_paused = False
        trainer_is_locked = False
        pause_pending = False
        resume_pending = False

        # Acessa comandos e trainer de forma segura (podem ser None)
        commands = self.training_commands  # Pode ser None se não estiver treinando
        trainer_instance = None  # Precisaria de acesso ao objeto trainer, o que não é direto aqui.
        # Vamos confiar nos comandos e callbacks.

        if commands:
            pause_pending = commands.is_pause_pending()
            resume_pending = commands.is_resume_pending()
            # Precisamos inferir o estado 'is_paused' e 'locked'.
            # Se a pausa foi pedida (pending) E NENHUM callback de início/fim de pausa ocorreu ainda, está "locked".
            # Se o callback on_pause_initiated ocorreu E on_resume_completed NÃO ocorreu, está "paused".
            # Isso é complexo de rastrear SÓ com comandos. Usaremos o estado do widget e callbacks.

        # Lógica simplificada inicial: Se pause foi pedido, marca ON e talvez disable. Se resume foi pedido, marca OFF.
        if pause_pending:
            self.ui_state["pause_training"] = True
            self.pause_switch_widget.select()
            self.pause_switch_widget.configure(state="disabled")  # Assume travado se pendente
            print("[UI Init] Pause request pending, setting switch ON and DISABLED.")
        elif resume_pending:  # Menos provável de acontecer no início, mas por segurança
            self.ui_state["pause_training"] = False
            self.pause_switch_widget.deselect()
            self.pause_switch_widget.configure(state="normal")
            print("[UI Init] Resume request pending, setting switch OFF and NORMAL.")
        else:  # Nenhuma requisição pendente, estado inicial normal
            self.ui_state["pause_training"] = False
            self.pause_switch_widget.deselect()
            self.pause_switch_widget.configure(state="normal")
            print("[UI Init] No pending requests, setting switch OFF and NORMAL.")

    # Em TrainUI.toggle_pause
    def toggle_pause(self):
        """Chamado quando o switch de pausa é clicado."""
        if not self.training_commands:
            print("[UI] Cannot pause/resume: Not training.")
            # Reverte visualmente o switch se não estiver treinando
            # Lê o valor ATUAL da variável antes de reverter
            current_val = self.pause_switch_var.get()
            self.pause_switch_var.set(not current_val)  # Inverte o valor na variável
            return

        # Obtém o estado DESEJADO pelo clique (o valor que a variável terá APÓS o clique)
        # A variável já foi atualizada pelo clique antes do comando ser chamado
        is_checked = self.pause_switch_var.get()
        print(f"[UI] Toggle Pause clicked. Switch variable is now: {is_checked}")

        if is_checked:  # Usuário quer PAUSAR (variável agora é True)
            print("[UI] Requesting pause...")
            success = self.training_commands.request_pause()
            if success:
                print("[UI] Pause request sent successfully. Waiting for trainer confirmation (callback).")
                # O callback handle_pause_request_accepted vai desabilitar o switch
            else:
                print("[UI] Pause request rejected by commands. Reverting switch variable.")
                # Reverte o estado da variável Tkinter
                self.pause_switch_var.set(False)
        else:  # Usuário quer RETOMAR (variável agora é False)
            print("[UI] Requesting resume...")
            success = self.training_commands.request_resume()
            if success:
                print("[UI] Resume request sent successfully. Waiting for trainer confirmation (callback).")
            else:
                print("[UI] Resume request rejected by commands. Reverting switch variable.")
                # Reverte o estado da variável Tkinter
                self.pause_switch_var.set(True)

    def handle_pause_request_accepted_threadsafe(self):
        print("[UI Callback Thread] Pause request accepted by trainer.")
        self.after(0, self._handle_pause_request_accepted_ui)

    def handle_pause_initiated_threadsafe(self):
        print("[UI Callback Thread] Pause initiated by trainer (model on CPU).")
        self.after(0, self._handle_pause_initiated_ui)

    def handle_resume_started_threadsafe(self):
        print("[UI Callback Thread] Resume started by trainer.")
        self.after(0, self._handle_resume_started_ui)

    def handle_resume_completed_threadsafe(self):
        print("[UI Callback Thread] Resume completed by trainer.")
        self.after(0, self._handle_resume_completed_ui)

    # --- Métodos que rodam na thread da UI (Chamados via self.after) ---
    # Em TrainUI._handle_pause_request_accepted_ui
    def _handle_pause_request_accepted_ui(self):
        print("[UI Thread] Updating UI for pause request accepted: Switch ON, DISABLED.")
        if self.pause_switch_widget:
            self.pause_switch_var.set(True)  # Confirma estado lógico/visual ON
            self.pause_switch_widget.configure(state="disabled")  # Trava

    # Em TrainUI._handle_pause_initiated_ui
    def _handle_pause_initiated_ui(self):
        print("[UI Thread] Updating UI for pause initiated: Switch ON, NORMAL (can resume).")
        if self.pause_switch_widget:
            self.pause_switch_var.set(True)  # Continua ON (está pausado)
            self.pause_switch_widget.configure(state="normal")  # Libera para clicar e retomar

    # Em TrainUI._handle_resume_started_ui
    def _handle_resume_started_ui(self):
        print("[UI Thread] Updating UI for resume started: Switch OFF, DISABLED (optional).")
        if self.pause_switch_widget:
            self.pause_switch_var.set(False)  # Estado lógico/visual vai pra OFF
            # Opcional: desabilitar enquanto move de volta pra GPU
            # self.pause_switch_widget.configure(state="disabled")

    # Em TrainUI._handle_resume_completed_ui
    def _handle_resume_completed_ui(self):
        print("[UI Thread] Updating UI for resume completed: Switch OFF, NORMAL.")
        if self.pause_switch_widget:
            self.pause_switch_var.set(False)  # Garante estado lógico/visual OFF
            self.pause_switch_widget.configure(state="normal")  # Garante habilitado

    def lora_tab(self, master) -> LoraTab:  # Note: Renomeado de create_lora_tab se necessário
        # A classe LoraTab já deve lidar com seu próprio 'master'
        lora_tab_instance = LoraTab(master, self.train_config, self.ui_state)
        # LoraTab provavelmente chama pack/grid internamente
        return lora_tab_instance  # Retorna a instância criada

    def embedding_tab(self, master):  # Note: Renomeado de create_embedding_tab se necessário
        # Conteúdo original do método, garantindo usar 'master'
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        components.label(frame, 0, 0, "Base embedding", tooltip="...")
        components.file_entry(frame, 0, 1, self.ui_state, "embedding.model_name", path_modifier=lambda x: ...)
        components.label(frame, 1, 0, "Token count", tooltip="...")
        components.entry(frame, 1, 1, self.ui_state, "embedding.token_count")
        components.label(frame, 2, 0, "Initial embedding text", tooltip="...")
        components.entry(frame, 2, 1, self.ui_state, "embedding.initial_embedding_text")
        components.label(frame, 3, 0, "Embedding Weight Data Type", tooltip="...")
        components.options_kv(
            frame,
            3,
            1,
            [("float32", DataType.FLOAT_32), ("bfloat16", DataType.BFLOAT_16)],
            self.ui_state,
            "embedding_weight_dtype",
        )
        components.label(frame, 4, 0, "Placeholder", tooltip="...")
        components.entry(frame, 4, 1, self.ui_state, "embedding.placeholder")
        components.label(frame, 5, 0, "Output embedding", tooltip="...")
        components.switch(frame, 5, 1, self.ui_state, "embedding.is_output_embedding")

        frame.pack(fill="both", expand=True)
        return frame  # Retorna o frame criado

    def create_additional_embeddings_tab(self, master) -> AdditionalEmbeddingsTab:
        # A classe AdditionalEmbeddingsTab já deve lidar com seu próprio 'master'
        add_emb_tab_instance = AdditionalEmbeddingsTab(master, self.train_config, self.ui_state)
        # AdditionalEmbeddingsTab provavelmente chama pack/grid internamente
        return add_emb_tab_instance  # Retorna a instância criada

    def create_tools_tab(self, master):
        # Conteúdo original do método, garantindo usar 'master'
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        components.label(frame, 0, 0, "Dataset Tools", tooltip="...")
        components.button(frame, 0, 1, "Open", self.open_dataset_tool)
        components.label(frame, 1, 0, "Convert Model Tools", tooltip="...")
        components.button(frame, 1, 1, "Open", self.open_convert_model_tool)
        components.label(frame, 2, 0, "Sampling Tool", tooltip="...")
        components.button(frame, 2, 1, "Open", self.open_sampling_tool)
        components.label(frame, 3, 0, "Profiling Tool", tooltip="...")
        components.button(frame, 3, 1, "Open", self.open_profiling_tool)

        frame.pack(fill="both", expand=True)
        return frame  # Retorna o frame criado

    def change_model_type(self, model_type: ModelType):
        # A lógica original já verifica se as abas existem (são None ou não)
        # Nenhuma alteração necessária aqui devido ao lazy loading
        if self.model_tab:  # Verifica se a aba já foi criada
            self.model_tab.refresh_ui()
        if self.training_tab:  # Verifica se a aba já foi criada
            self.training_tab.refresh_ui()
        # O lora_tab original não existia aqui, mas se existisse, seria:
        # if self.lora_tab_content:
        #    self.lora_tab_content.refresh_ui()

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return  # Segurança

        # START: Otimização - Modificar apenas a *existência* da aba no tabview
        # A criação do *conteúdo* será tratada pelo _on_tab_change quando/se selecionada.

        # Lógica para remover abas se o método não for LORA/Embedding
        if training_method != TrainingMethod.LORA and "LoRA" in self.tabview._tab_dict:
            print("[UI] Removing LoRA tab structure.")
            self.tabview.delete("LoRA")
            self.lora_tab_content = None  # Garante que será recriado se necessário
        if training_method != TrainingMethod.EMBEDDING and "embedding" in self.tabview._tab_dict:
            print("[UI] Removing Embedding tab structure.")
            self.tabview.delete("embedding")
            self.embedding_tab_content = None  # Garante que será recriado

        # Lógica para adicionar abas se o método for LORA/Embedding e a aba não existir
        if training_method == TrainingMethod.LORA and "LoRA" not in self.tabview._tab_dict:
            print("[UI] Adding LoRA tab structure.")
            self.tabview.add("LoRA")
            # NÃO cria o conteúdo aqui, _on_tab_change fará isso
        if training_method == TrainingMethod.EMBEDDING and "embedding" not in self.tabview._tab_dict:
            print("[UI] Adding Embedding tab structure.")
            self.tabview.add("embedding")
            # NÃO cria o conteúdo aqui, _on_tab_change fará isso

        # Atualiza a aba de modelo, se já tiver sido criada
        if self.model_tab:
            self.model_tab.refresh_ui()
        # END: Otimização - Modificar apenas a *existência* da aba no tabview

    def load_preset(self):
        # A lógica original já verifica se a aba existe
        if not self.tabview:
            return

        if self.additional_embeddings_tab:  # Verifica se já foi criada
            self.additional_embeddings_tab.refresh_ui()
        # Se outras abas precisassem de refresh no load_preset, adicionar verificações similares:
        # if self.general_tab: self.general_tab...
        # if self.model_tab: self.model_tab... etc.

    # Métodos restantes (open_tensorboard, callbacks, tools, training logic, export, etc.)
    # geralmente não precisam de alteração para o lazy loading das abas,
    # pois interagem com a lógica de treino ou janelas separadas, ou
    # já possuem verificações implícitas (e.g., `self.training_commands` sendo None).

    def open_tensorboard(self):
        webbrowser.open("http://localhost:" + str(self.train_config.tensorboard_port), new=0, autoraise=False)

    def on_update_train_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        self.set_step_progress(train_progress.epoch_step, max_sample)
        self.set_epoch_progress(train_progress.epoch, max_epoch)

    def on_update_status(self, status: str):
        if self.status_label:  # Adiciona verificação
            self.status_label.configure(text=status)

    def open_dataset_tool(self):
        from modules.ui.CaptionUI import CaptionUI

        window = CaptionUI(self, None, False)
        self.wait_window(window)

    def open_convert_model_tool(self):
        window = ConvertModelUI(self)
        self.wait_window(window)

    def open_sampling_tool(self):
        if not self.training_callbacks and not self.training_commands:
            window = SampleWindow(
                self,
                train_config=self.train_config,
            )
            self.wait_window(window)
            torch_gc()

    def open_profiling_tool(self):
        self.profiling_window.deiconify()

    def open_sample_ui(self):
        training_callbacks = self.training_callbacks
        training_commands = self.training_commands

        if training_callbacks and training_commands:
            window = SampleWindow(
                self,
                callbacks=training_callbacks,
                commands=training_commands,
            )
            self.wait_window(window)
            if training_callbacks:  # Adiciona verificação
                training_callbacks.set_on_sample_custom()

    def __training_thread_function(self):
        from modules.trainer.GenericTrainer import GenericTrainer

        error_caught = False
        trainer = None  # Inicializa para garantir que o del funcione no finally

        try:
            self.training_callbacks = TrainCallbacks(
                on_update_train_progress=self.on_update_train_progress,
                on_update_status=self.on_update_status,
                on_pause_request_accepted=self.handle_pause_request_accepted_threadsafe,
                on_pause_initiated=self.handle_pause_initiated_threadsafe,
                on_resume_started=self.handle_resume_started_threadsafe,
                on_resume_completed=self.handle_resume_completed_threadsafe,
            )

            # Garante que training_commands existe antes de passar
            if not self.training_commands:
                self.training_commands = TrainCommands()  # Cria se não existir (embora start_training deva criar)
                print("Warn: Training commands were None in thread function, created new.")

            if self.train_config.cloud.enabled:
                from modules.trainer.CloudTrainer import CloudTrainer

                # Garante que self.cloud_tab (e seu reattach) exista se cloud estiver habilitada
                # Se cloud pode ser habilitada sem a aba ser visível, pode precisar forçar a criação
                if not self.cloud_tab and "cloud" in self.tabview._name_list:
                    print("[UI] Forcing lazy load of Cloud tab for CloudTrainer.")
                    self._on_tab_change()  # Força o carregamento se a aba existir mas não foi clicada

                reattach_val = self.cloud_tab.reattach if self.cloud_tab else False  # Default seguro
                trainer = CloudTrainer(
                    self.train_config, self.training_callbacks, self.training_commands, reattach=reattach_val
                )
            else:
                ZLUDA.initialize_devices(self.train_config)
                trainer = GenericTrainer(self.train_config, self.training_callbacks, self.training_commands)

            trainer.start()
            if self.train_config.cloud.enabled:
                # UIState pode não estar totalmente sincronizado se a aba cloud não foi vista
                # Idealmente, a config passada para o trainer é a fonte da verdade
                pass  # A lógica de secrets pode precisar de revisão se depender do UIState atualizado pela UI
            trainer.train()

        except Exception:
            error_caught = True
            traceback.print_exc()
            self.on_update_status("Error: Check console")  # Atualiza status no erro
        finally:
            # Bloco finally garante a limpeza
            if trainer:
                if self.train_config.cloud.enabled and hasattr(self.train_config, "secrets"):
                    # Tenta atualizar secrets mesmo em caso de erro ou sucesso
                    try:
                        self.ui_state.get_var("secrets.cloud").update(self.train_config.secrets.cloud)
                    except Exception as e:
                        print(f"Warn: Failed to update cloud secrets state after training: {e}")

                trainer.end()
                del trainer  # Libera referência

            self.training_thread = None
            # self.training_commands = None # Mantém os comandos? Ou reseta? Depende da lógica de stop/restart
            torch.clear_autocast_cache()
            torch_gc()

            if not error_caught:
                self.on_update_status("Stopped")

            # Garante que o botão seja reativado na thread principal (usando self.after)
            self.after(0, self._reset_training_button)

    def _reset_training_button(self):
        """Helper para resetar os botões e switches de controle na thread principal."""
        print("[UI Thread] Resetting training control buttons/switches.")
        if self.training_button:
            self.training_button.configure(text="Start Training", state="normal")
            # Se o estado for 'stopping' (disabled), força para 'normal'
            if self.training_button.cget("state") == "disabled":
                self.training_button.configure(state="normal")

        # Reseta o switch de pausa para OFF e NORMAL
        if self.pause_switch_widget:
            self.pause_switch_var.set(False)
            self.pause_switch_widget.configure(state="normal")

        # Reseta os comandos AQUI para garantir que não haja comandos pendentes após parada/erro
        # Isso evita que um pause_request antigo seja processado se o treino for reiniciado
        self.training_commands = None
        self.training_callbacks = None  # Limpa callbacks também

    def start_training(self):
        if self.training_thread is None:
            self.top_bar_component.save_default()

            self.training_commands = TrainCommands()  # Cria novos comandos para a sessão

            # --- CORREÇÃO AQUI ---
            # Reseta o estado do switch de pausa usando a variável dedicada
            if self.pause_switch_var: # Checa se a variável existe
                self.pause_switch_var.set(False) # Define a variável como False (desligado)
            # A atualização visual (deselect, configure) pode ser feita aqui ou confiar no _reset_training_button
            # Para garantir, vamos fazer aqui também:
            if self.pause_switch_widget:
                 self.pause_switch_widget.deselect()
                 self.pause_switch_widget.configure(state="normal")
            # --- FIM DA CORREÇÃO ---

            if self.training_button:  # Verifica se existe
                self.training_button.configure(text="Stop Training", state="normal")

            self.training_thread = threading.Thread(
                target=self.__training_thread_function, daemon=True
            )  # Use daemon=True?
            self.training_thread.start()
        elif self.training_commands:  # Verifica se comandos existem para parar
            if self.training_button:  # Verifica se existe
                # Muda o texto para indicar que está parando e desabilita
                self.training_button.configure(text="Stopping...", state="disabled")
            self.after(0, lambda: self.on_update_status("Stopping..."))  # Atualiza status via after
            self.training_commands.stop()  # Envia o comando de parada
        else:
            print("Warn: Stop training called but no training commands object exists.")
            # Possivelmente resetar o botão se estiver em estado inconsistente
            self._reset_training_button()  # Usa o método de reset

    def export_training(self):
        file_path = filedialog.asksaveasfilename(
            filetypes=[("JSON config", "*.json"), ("All Files", "*.*")],
            initialdir=".",
            initialfile="config.json",
            defaultextension=".json",  # Adiciona extensão padrão
        )

        if file_path:
            try:
                # Usar secrets=False diretamente no to_pack_dict
                config_dict = self.train_config.to_pack_dict(secrets=False)
                with open(file_path, "w", encoding="utf-8") as f:  # Especifica encoding
                    json.dump(config_dict, f, indent=4)
                self.on_update_status(f"Config exported to {Path(file_path).name}")
            except Exception as e:
                traceback.print_exc()
                self.on_update_status(f"Error exporting config: {e}")

    # Funções sample_now, backup_now, save_now não precisam mudar
    # Elas dependem de self.training_commands que é gerenciado pelo start/stop
    def sample_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.sample_default()
        else:
            self.on_update_status("Cannot sample: Not training")

    def backup_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.backup()
        else:
            self.on_update_status("Cannot backup: Not training")

    def save_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.save()
        else:
            self.on_update_status("Cannot save: Not training")


# Fim da classe TrainUI
