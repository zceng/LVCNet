
import torch 

from .stft_loss import * 

class PWGLoss(torch.nn.Module):

    def __init__(self, stft_loss_params={}):
        super(PWGLoss, self).__init__() 
        self.stft_criterion = MultiResolutionSTFTLoss( **stft_loss_params )
        self.mse_criterion = torch.nn.MSELoss() 


    def stft_loss(self, audio, audio_):
        sc_loss, mag_loss = self.stft_criterion( audio_.squeeze(1), audio.squeeze(1) ) 
        return sc_loss, mag_loss 

    def adversarial_loss(self, prob_ ):
        return self.mse_criterion( prob_, torch.ones_like( prob_ ) ) 

    def discriminator_loss(self, prob, prob_ ):
        real_loss = self.mse_criterion( prob,  torch.ones_like( prob ) ) 
        fake_loss = self.mse_criterion( prob_, torch.zeros_like(prob_) )
        return real_loss, fake_loss
