from typing import Union

from .pwg_loss import PWGLoss

loss_modules = {
    "PWGLoss": PWGLoss
}

def create_loss(name, params, device) -> Union[PWGLoss]:
    return loss_modules[ name ]( **params ).to(device)