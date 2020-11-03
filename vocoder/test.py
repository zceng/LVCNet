
import argparse, yaml, datetime, os, time 
import yaml, tqdm 
from collections import defaultdict
import soundfile

import torch 
from vocoder.datasets import create_dataloader 
from vocoder.models import create_model
from vocoder.strategy import create_strategy
from vocoder.utils.log import Logger
from vocoder.hparams import Hyperparameter 



class Tester:

    def __init__(self, args, hparams: Hyperparameter):
        self.log = Logger(args.exp_dir, tensorboard=False) 

        self.exp_dir = args.exp_dir 
        self.device = torch.device( args.device )
        self.hparams = hparams 
        
        self.model     = create_model( hparams.model_name, hparams.model_params, device=self.device )
        self.strategy  = create_strategy( hparams.strategy_name, hparams.strategy_params ) 
        self.restore_checkpoint()

        self.test_result_dir = os.path.join( self.exp_dir, f'test-{self.step}-step' )
        os.makedirs( self.test_result_dir, exist_ok=True )

        self.train_results = defaultdict(float) 

    def restore_checkpoint(self):
        checkpoint = os.readlink( os.path.join( self.exp_dir, 'checkpoint.pt') )
        state_dict = torch.load( checkpoint, map_location='cpu')  
        self.step = state_dict['step']
        self.model.load_state_dict( state_dict['model'] ) 
        self.log.info( f"Restore model from {checkpoint}" )

    def init_dataloader(self):
        ''' initialize dataloader for training and evaluate '''
        dataset_config = {
            'metadata_file': self.hparams.test_metadata_file,
            'hop_length': self.hparams.hop_length,
            'batch_mel_length': self.hparams.batch_mel_length,
            'cut': False
        }
        self.dataloader = create_dataloader( 
                            dataset_classname=self.hparams.dataset_classname, 
                            dataset_config=dataset_config,
                            batch_size=1,
                            num_workers=self.hparams.dataset_num_workers,
                            shuffle=False,
                            drop_last=False )

    def run(self):
        # 初始化 dataloader 
        self.init_dataloader() 
        total_rtf = 0.0
        with tqdm.tqdm( self.dataloader, desc= "Test" ) as phbar:
            for idx, batch in enumerate( phbar, start=1 ): 
                st = time.time() 
                result = self.strategy.test_step( batch, self.model) 
                tc = time.time() - st

                audio = result['audio'].squeeze(0).squeeze(0).cpu().numpy() 
                soundfile.write(os.path.join( self.test_result_dir, f"{idx:04d}_gene.wav"),
                        audio, self.hparams.sample_rate, "PCM_16")
                real_audio = batch[0].squeeze(0).squeeze(0).numpy()
                soundfile.write(os.path.join( self.test_result_dir, f"{idx:04d}_real.wav"),
                        real_audio, self.hparams.sample_rate, "PCM_16") 

                rtf = tc*self.hparams.sample_rate/len(audio)
                total_rtf += rtf
                phbar.set_postfix({"RTF": rtf})

        self.log.info('Average RTF: {}'.format( total_rtf/idx ) )
        self.log.info( f'Test result saving into {self.test_result_dir}' )


def main():
    parser = argparse.ArgumentParser(
        description="Train LVC-WaveGAN (See detail in vocoder/train.py).")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--exp-dir", type=str, required=True,
                        help="the directory saving expriment data, "
                        "including model checkpoints, log, results. ")
    parser.add_argument("--device", default='cuda', type=str,
                        help="the device for training. (default: cuda:0)")
    args = parser.parse_args() 
    hparams = Hyperparameter( args.config ) 

    tester = Tester(args, hparams)

    try:
        tester.run() 
    except KeyboardInterrupt:
        pass 


if __name__ == "__main__":
    main() 