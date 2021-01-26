from .base_params import BaseParameters


class Parameters(BaseParameters):
    def __init__(self):
        super().__init__()

        self.seed = 1234
        self.save_path = "."

        self.img_shape = (256, 256)
        self.transforms = ["rotate", "flip", "translate"]

        self.batchsize = 32
        self.num_classes = 10
        self.arch = "densenet201"
        self.model_path = "checkpoints/train_hotness_bins_all_densenet201_medical"

        self.num_iter = 10
