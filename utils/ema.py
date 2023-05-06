import torch
import torch.nn as nn

def ema_update(ema_model, model, decay_rate=0.995):
        """
        Exponential model weight averaging update.
        """
        for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
            p_ema.data *= decay_rate
            p_ema.data += p_model.data * (1 - decay_rate)

@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked            