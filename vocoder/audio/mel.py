import torch
import numpy as np

from .stft import TacotronSTFT
from .util import griffin_lim


def griffin_lim_inverse_mel(mel, tacotron_stft, griffin_iters=60):
    ''' generate waveform using griffin_lim according to the mel-spectrums.  

    Args: 
        mel (Tensor): shape (B, L, C) 
        tacotron_stft (TacotronSTFT): A transformation class to calcuate the mel-spectrum 
        griffin_iters (int): the iters for griffin_lim  
    
    Returns:
        audio (Tensor): the generated waveform. 
    '''
    # mel = torch.stack([torch.from_numpy(_denormalize(mel.numpy()))])
    mel_decompress = tacotron_stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(
        spec_from_mel[:, :, :-1]), tacotron_stft.stft_fn, griffin_iters)

    return audio 
