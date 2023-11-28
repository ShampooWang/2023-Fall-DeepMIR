from torch import nn
from pytorch_lightning.callbacks import Callback

__all__ = ["freeze_model", "unfreeze_model", "compute_grad_norm", "GradNormCallback"]

def freeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True

def compute_grad_norm(model: nn.Module) -> int:
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1/2)
    return total_norm

class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """
    def on_after_backward(self, trainer, model):
        print(f"my_model/grad_norm: {compute_grad_norm(model)}")