
from typing import Union

from .parallel_wavegan import ParallelWaveGAN 
from .lvcgan import LVCNetWaveGAN

model_list = {
    "ParallelWaveGAN": ParallelWaveGAN, 
    "LVCNetWaveGAN": LVCNetWaveGAN
}


def create_model(name, params, device) -> Union[ParallelWaveGAN]:
    ''' Create model according to the model classname  
    Args:
        name (str): model classname.  
        params (dict): the parameter for create model.
    Return:
        torch.nn.Module : Model. 
    '''
    return model_list[ name ](**params).to(device) 

