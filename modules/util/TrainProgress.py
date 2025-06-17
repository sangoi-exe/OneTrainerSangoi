class TrainProgress:
    def __init__(
            self,
            epoch: int = 0,
            epoch_step: int = 0,
            epoch_sample: int = 0,
            global_step: int = 0,
            total_steps: int = 0,
            steps_per_epoch: int = 0,
    ):
        self.epoch = epoch
        self.epoch_step = epoch_step
        self.epoch_sample = epoch_sample
        self.global_step = global_step
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch

    def next_step(self, batch_size: int):
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1

    def next_epoch(self):
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1
        
    def set_total_steps(self, total_steps):
        self.total_steps = total_steps
        
    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch

    def filename_string(self):
        return f"{self.global_step}-{self.epoch}-{self.epoch_step}"
