import os 
import numpy as np 
from scipy.io import wavfile

import torch
from torch.utils.data import DataLoader, Dataset  

from vocoder.audio import load_wav_to_torch 
from .utils import read_metadata 

class PWGAudioMelNoiseDataset(Dataset):
    ''' the Pytorch Dataset for loading audio(.wav) and mel(.npy) '''

    def __init__(self, metadata_file, batch_mel_length, sample_rate, hop_length, cut=True):
        '''Initialize   
        Args:
            metadata_file (str): the file including paths of audio and mel.  
            batch_mel_length (int): the length of mel-spectrum for batch. 
            hop_length (int): the hop length used when calculating mel-spectrum.

        Description:
            Example of metadata_file:
                ./data/wavs/001.wav|./temp/mels/001.npy 
                ./data/wavs/002.wav|./temp/mels/002.npy
                ./data/wavs/003.wav|./temp/mels/003.npy     
        '''
        super().__init__() 
        self.batch_mel_length = batch_mel_length 
        self.hop_length = hop_length
        self.sample_rate = sample_rate 
        self.cut = cut

        self.metadata = read_metadata( metadata_file ) 
        # metadata: contains paths of entire wav files and mel-spectrum files. 
        #   Examples: [ ('./data/wavs/001.wav', './dump/mels/001.npy'), ... ] 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        '''
        Returns: 
            Tensor (float): audio, shape (L,)
            Tensor (float): mel-spectrum, shape ( ML, MC) 
            Tensor (float): guassian noise with the same shape as audio, shape (L,) 
        
        Note: 
            the length of mel-spectrum (ML) is equal to `batch_mel_length`
            the equation relationship between the length of audio and mel-spectrum: 
                L = ML * hop_length
        '''
        wav_path, mel_path = self.metadata[ idx ] 
        
        audio, sr = load_wav_to_torch( wav_path, self.sample_rate )
        assert sr == self.sample_rate 

        mel = np.load( mel_path ) 

        if self.cut:
            assert mel.shape[0] > self.batch_mel_length + 1, f"the length of audio is too short: {wav_path}" 
            mel_start = np.random.randint( 0, mel.shape[0] - self.batch_mel_length - 1 ) 
            audio_start = (mel_start + 2) * self.hop_length 

            mel = mel[ mel_start : mel_start + self.batch_mel_length ] 
            audio = audio[ :, audio_start : audio_start + (self.batch_mel_length - 4) * self.hop_length ] 
        else:
            audio = audio[ :, 2*self.hop_length:(mel.shape[0] - 2) * self.hop_length ] 

        mel = torch.from_numpy( mel.T ) 
        noise = torch.randn_like( audio  )
        return audio, mel, noise 




        



