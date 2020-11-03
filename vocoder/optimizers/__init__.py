

from typing import Union

from .pwg_opt import PWGOptimizer

optimizer_list = {
    "PWGOptimizer": PWGOptimizer
}

def create_optimizer(name, model, params) -> Union[PWGOptimizer]:
    return optimizer_list[ name ]( model, **params ) 
