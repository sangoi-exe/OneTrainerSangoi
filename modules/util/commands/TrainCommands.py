
import threading
from modules.util.config.SampleConfig import SampleConfig


class TrainCommands:
    def __init__(
            self,
            on_command=None# Callable[[TrainCommands], None] = lambda _: None
    ):
        self.reset()
        self.__stop_command = False
        self.__on_command = on_command
        self.__pause_requested = False
        self.__resume_requested = False
        self.__command_lock = threading.Lock() # Para segurança em ambientes multithread (mesmo que a UI seja single)

    def reset(self):
        #don't reset stop
        self.__sample_custom_commands = []
        self.__sample_default_command = False
        self.__backup_command = False
        self.__save_command = False

    def set_on_command(
            self,
            on_command#: Callable[[TrainCommands], None] = lambda _: None
    ):
        self.__on_command = on_command

    def get_and_reset_on_command(self):
        on_command = self.__on_command
        self.__on_command=None
        return on_command

    def stop(self):
        self.__stop_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_stop_command(self) -> bool:
        with self.__command_lock:
            return self.__stop_command

    def sample_custom(self, sample_params: SampleConfig):
        self.__sample_custom_commands.append(sample_params)
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_sample_custom_commands(self) -> list[SampleConfig]:
        with self.__command_lock:
            sample_custom_commands = self.__sample_custom_commands
            self.__sample_custom_commands = []
            return sample_custom_commands

    def sample_default(self):
        self.__sample_default_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_sample_default_command(self) -> bool:
         with self.__command_lock:
            sample_default_command = self.__sample_default_command
            self.__sample_default_command = False
            return sample_default_command

    def backup(self):
        self.__backup_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_backup_command(self) -> bool:
        with self.__command_lock:
            backup_command = self.__backup_command
            self.__backup_command = False
            return backup_command

    def save(self):
        self.__save_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_save_command(self) -> bool:
        with self.__command_lock:
            save_command = self.__save_command
            self.__save_command = False
            return save_command

    def request_pause(self):
        with self.__command_lock:
            # Só permite solicitar pausa se não estiver já solicitada ou resumindo
            if not self.__pause_requested and not self.__resume_requested:
                self.__pause_requested = True
                print("[Commands] Pause Requested") # Log
                if self.__on_command:
                    self.__on_command(self)
                return True # Indica que a requisição foi aceita
            return False # Indica que a requisição foi ignorada (já pendente)

    def request_resume(self):
        with self.__command_lock:
            # Só permite solicitar resume se não estiver já solicitado ou pausando
            if not self.__resume_requested and not self.__pause_requested:
                self.__resume_requested = True
                print("[Commands] Resume Requested") # Log
                if self.__on_command:
                    self.__on_command(self)
                return True # Indica que a requisição foi aceita
            return False # Indica que a requisição foi ignorada (já pendente)

    def get_and_reset_pause_request(self) -> bool:
        with self.__command_lock:
            pause_req = self.__pause_requested
            if pause_req:
                self.__pause_requested = False # Reseta a flag
                # Importante: Não resetar resume_requested aqui
            return pause_req

    def get_and_reset_resume_request(self) -> bool:
        with self.__command_lock:
            resume_req = self.__resume_requested
            if resume_req:
                self.__resume_requested = False # Reseta a flag
                # Importante: Não resetar pause_requested aqui
            return resume_req
    
    def is_pause_pending(self) -> bool:
        with self.__command_lock:
            return self.__pause_requested

    def is_resume_pending(self) -> bool:
        with self.__command_lock:
            return self.__resume_requested
