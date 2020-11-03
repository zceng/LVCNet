
import torch 

from vocoder.models import ParallelWaveGAN
from .radam import RAdam


class PWGOptimizer:

    def __init__(self, model: ParallelWaveGAN,
                 generator_optimizer_params={"lr": 1e-4, "eps": 1e-6},
                 generator_scheduler_params={"step_size": 200000, "gamma": 0.5},
                 discriminator_optimizer_params={"lr": 5e-5, "eps": 1e-6},
                 discriminator_scheduler_params={"step_size": 200000, "gamma": 0.5}):
        self.generator_optimizer = RAdam( 
            model.generator.parameters(), **generator_optimizer_params ) 
        self.generator_scheduler = torch.optim.lr_scheduler.StepLR( 
            optimizer=self.generator_optimizer, **generator_scheduler_params)

        self.discriminator_optimizer = RAdam( 
            model.discriminator.parameters(), **discriminator_optimizer_params ) 
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.discriminator_optimizer, **discriminator_scheduler_params) 

    def state_dict(self):
        return {
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "generator_scheduler": self.generator_scheduler.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "discriminator_scheduler": self.discriminator_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.generator_optimizer.load_state_dict( state_dict["generator_optimizer"] )
        self.generator_scheduler.load_state_dict( state_dict["generator_scheduler"] )
        self.discriminator_optimizer.load_state_dict( state_dict["discriminator_optimizer"] )
        self.discriminator_scheduler.load_state_dict( state_dict["discriminator_scheduler"] )



