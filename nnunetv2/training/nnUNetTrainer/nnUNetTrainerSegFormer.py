from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.segformer3d_wrapper import SegFormer3D_Wrapper
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch

class nnUNetTrainerSegFormer3D(nnUNetTrainer):
    """
    Drop-in replacement for the default trainer.
    We simply override build_network().
    """

    def build_network(self):
        # ---- gather info from the plan that nnUNet already parsed ----
        num_input_channels = self.plans['num_modalities']  # e.g. 1 or 4
        num_classes        = self.label_manager.num_segments()

        network = SegFormer3D_Wrapper(
            in_channels=num_input_channels,
            n_classes=num_classes,
            deep_supervision=self.configuration.network_params.use_deep_supervision,
        )
        # Use checkpoint-saving & AMP settings from parent
        network = network.to(self.device, dtype=torch.float16 if self.compute_dtype == torch.float16 else torch.float32)
        return network
