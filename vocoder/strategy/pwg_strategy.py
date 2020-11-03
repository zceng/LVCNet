

import torch

from vocoder.models import ParallelWaveGAN
from vocoder.losses import PWGLoss
from vocoder.optimizers import PWGOptimizer
from .base import TrainStrategy




class PWGStrategy(TrainStrategy):

    def __init__(self, 
                 lambda_adv=4.0,
                 discriminator_start_steps=100000,
                 generator_grad_norm=10,
                 discriminator_grad_norm=1):
        super().__init__() 

        self.lambda_adv = lambda_adv
        self.discriminator_start_steps = discriminator_start_steps
        self.generator_grad_norm = generator_grad_norm
        self.discriminator_grad_norm = discriminator_grad_norm

    def train_step(self, batch, step, 
                   model: ParallelWaveGAN, 
                   loss: PWGLoss, 
                   optimizer: PWGOptimizer):
        '''Train strategy for Parallel WaveGAN.
        Args:
            batch (list): the batch data for training model. 
                           [ audio(B,L), mel(B,ML,MC), noise(B,L) ]  
            step (int): current global step in training process. 
            model (ParallelWaveGAN): the parallel wavegan model.
            loss (PWGLoss): the loss module for parallel wavegan
            optimizer (PWGOptimizer): customized optimizer. 
        Returns:
            dict: the loss value dict.
        '''
        device = next(model.parameters()).device
        audio, mel, noise = [ x.to(device) for x in batch ]

        #######################
        #      Generator      #
        #######################
        audio_ = model.generator(noise, mel) 

        sc_loss, mag_loss = loss.stft_loss( audio, audio_ ) 
        gen_loss = sc_loss + mag_loss

        adv_loss = torch.zeros(1)
        if step > self.discriminator_start_steps:
            prob_ = model.discriminator( audio_ ) 
            adv_loss = loss.adversarial_loss( prob_ ) 
            gen_loss += self.lambda_adv * adv_loss 
        
        optimizer.generator_optimizer.zero_grad() 
        gen_loss.backward() 
        if self.generator_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.generator.parameters(), 
                self.generator_grad_norm) 
        optimizer.generator_optimizer.step() 
        optimizer.generator_scheduler.step()

        #######################
        #    Discriminator    #
        #######################
        real_loss, fake_loss, disc_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        if step > self.discriminator_start_steps:
            with torch.no_grad():
                audio_ = model.generator( noise, mel ) 
            prob  = model.discriminator( audio ) 
            prob_ = model.discriminator( audio_.detach() ) 

            real_loss, fake_loss = loss.discriminator_loss( prob, prob_ ) 
            disc_loss = real_loss + fake_loss 

            optimizer.discriminator_optimizer.zero_grad()
            disc_loss.backward() 
            if self.discriminator_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.discriminator.parameters(), 
                    self.discriminator_grad_norm) 
            optimizer.discriminator_optimizer.step()
            optimizer.discriminator_scheduler.step() 

        return { 
            "generator_loss": gen_loss.item(), 
            "spectral_convergence_loss": sc_loss.item(), 
            "log_stft_magnitude_loss": mag_loss.item(),
            "adversarial_loss": adv_loss.item(), 
            "discriminator_loss": disc_loss.item(), 
            "real_loss": real_loss.item(), 
            "fake_loss": fake_loss.item()
        }

    @torch.no_grad()
    def eval_step(self, batch, 
                  model: ParallelWaveGAN, 
                  loss: PWGLoss):
        device = next(model.parameters()).device
        audio, mel, noise = [ x.to(device) for x in batch ]

        audio_ = model.generator( noise, mel ) 
        prob_  = model.discriminator( audio_ ) 
        prob   = model.discriminator( audio ) 

        sc_loss, mag_loss = loss.stft_loss( audio, audio_ ) 
        adv_loss = loss.adversarial_loss( prob_ ) 
        gen_loss = sc_loss + mag_loss + self.lambda_adv * adv_loss 

        real_loss, fake_loss = loss.discriminator_loss( prob, prob_ ) 
        disc_loss = real_loss + fake_loss 

        return { 
            "generator_loss": gen_loss.item(), 
            "spectral_convergence_loss": sc_loss.item(), 
            "log_stft_magnitude_loss": mag_loss.item(),
            "adversarial_loss": adv_loss.item(), 
            "discriminator_loss": disc_loss.item(), 
            "real_loss": real_loss.item(), 
            "fake_loss": fake_loss.item()
        }
    
    @torch.no_grad()
    def test_step(self, batch, model: ParallelWaveGAN):
        device = next(model.parameters()).device
        audio, mel, noise = [ x.to(device) for x in batch ]

        audio_ = model.generator( noise, mel ) 
        return { 'audio' : audio_ }

        








