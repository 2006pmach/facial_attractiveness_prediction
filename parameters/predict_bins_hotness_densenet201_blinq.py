from .base_params import BaseParameters


class Parameters(BaseParameters):
    def __init__(self):
        super().__init__()

        self.seed = 1234
        self.save_path = "."

        self.img_shape = (256, 256)

        self.num_classes = 10
        self.arch = "densenet201"
        self.model_path = "checkpoints/train_hotness_bins_female_densenet201_blinq"

        self.num_iter = 1
