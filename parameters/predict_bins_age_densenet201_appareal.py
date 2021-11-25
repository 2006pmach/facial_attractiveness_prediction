from .base_params import BaseParameters


class Parameters(BaseParameters):
    def __init__(self):
        super().__init__()

        self.seed = 1234
        self.save_path = "."

        self.img_shape = (256, 256)

        self.num_classes = 101
        self.arch = "densenet201"
        self.model_path = "checkpoints/train_age_bins_all_densenet201_appareal"

        self.num_iter = 1
