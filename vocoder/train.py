
import argparse, yaml, datetime, os 
import yaml, tqdm 
from collections import defaultdict

import torch 
from vocoder.datasets import create_dataloader 
from vocoder.models import create_model
from vocoder.losses import create_loss 
from vocoder.optimizers import create_optimizer
from vocoder.strategy import create_strategy
from vocoder.utils.log import Logger
from vocoder.hparams import Hyperparameter 



class Trainer:

    def __init__(self, args, hparams: Hyperparameter):
        self.log = Logger(args.exp_dir) 

        self.exp_dir = args.exp_dir
        os.makedirs( self.exp_dir, exist_ok=True )

        self.device = torch.device( args.device )
        self.hparams = hparams 
        
        self.step = 1 
        self.epoch = 1
        
        self.model     = create_model( hparams.model_name, hparams.model_params, device=self.device )
        self.loss      = create_loss( hparams.loss_name, hparams.loss_params, device=self.device )
        self.optimizer = create_optimizer( hparams.opt_name, self.model, hparams.opt_params ) 
        self.strategy  = create_strategy( hparams.strategy_name, hparams.strategy_params ) 

        self.restore_checkpoint(args.restart, args.checkpoint)

        self.train_results = defaultdict(float) 

    def restore_checkpoint(self, restart=False, checkpoint=None):
        if not restart:
            try:
                pt = os.path.join( self.exp_dir, 'checkpoint.pt') 
                if checkpoint is None and os.path.islink(pt):
                    checkpoint = os.readlink(pt)
                if not os.path.isfile( checkpoint ):
                    print('start new training.')
                    return
                state_dict = torch.load( checkpoint, map_location='cpu')
                self.step = state_dict['step'] 
                self.epoch = state_dict['epoch'] 
                self.model.load_state_dict( state_dict['model'] ) 
                self.optimizer.load_state_dict( state_dict['optimizer'] )
                self.log.info( f"Restore model from {checkpoint}")
            except:
                print('Error in restore model. Start New training')

    def save_checkpoint(self):
        state_dict = {
            "step": self.step,
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict(), 
            "model": self.model.state_dict()
        }
        save_path = os.path.join( self.exp_dir, f'checkpoint-{self.step}.pt')
        link_path = os.path.join( self.exp_dir, 'checkpoint.pt')
        torch.save( state_dict, save_path )
        if os.path.islink(link_path):
            os.unlink(link_path)
        os.symlink(save_path, link_path)
        self.log.info( f'Save chechpoint as {save_path}' )

    def init_dataloader(self):
        ''' initialize dataloader for training and evaluate '''
        train_dataset_config = {
            'metadata_file': self.hparams.train_metadata_file,
            'hop_length': self.hparams.hop_length,
            'batch_mel_length': self.hparams.batch_mel_length
        }
        eval_dataset_config = {
            'metadata_file': self.hparams.eval_metadata_file,
            'hop_length': self.hparams.hop_length,
            'batch_mel_length': self.hparams.batch_mel_length
        }
        self.dataloader = {
            "train": create_dataloader( 
                            dataset_classname=self.hparams.dataset_classname, 
                            dataset_config=train_dataset_config,
                            batch_size=self.hparams.train_batch_size,
                            num_workers=self.hparams.dataset_num_workers,
                            shuffle=True,
                            drop_last=True ), 
            "eval": create_dataloader( 
                            dataset_classname=self.hparams.dataset_classname, 
                            dataset_config=eval_dataset_config,
                            batch_size=self.hparams.train_batch_size,
                            num_workers=1,
                            shuffle=False,
                            drop_last=False )
        }

    def train(self):
        # 初始化 dataloader 
        self.init_dataloader() 
        while True:
            with tqdm.tqdm( self.dataloader["train"], desc= f"Train, Epoch: {self.epoch}" ) as tqbar:
                for batch in tqbar: 
                    if self.step > self.hparams.max_train_steps:
                        return 
                    tqbar.set_postfix({"Step": self.step})
                    
                    result = self.strategy.train_step( batch, self.step, self.model, self.loss, self.optimizer ) 

                    self._check_log(result)
                    self._check_evaluate()

                    self.step += 1
                self.epoch += 1

    def evaluate(self):
        eval_results = defaultdict(float)
        for batch in tqdm.tqdm(self.dataloader["eval"], desc= f"Evaluate"):
            result = self.strategy.eval_step( batch, self.model, self.loss ) 
            for k in result:
                eval_results[k] += result[k] 

        self.log.info( f'Step {self.step}, Evaluate results:')
        for k in eval_results:
            v = eval_results[k] / len( self.dataloader["eval"] )
            self.log.add_scalar( f'evaluate/{k}', v, self.step) 
            self.log.info( f'  {k}: {v:.4f}' )
        
        self.log.flush()

    def _check_evaluate(self):
        if self.step % self.hparams.eval_interval_steps == 0:
            self.model.eval()
            self.evaluate()
            self.model.train()

    def _check_log(self, train_result):
        for k in train_result:
            self.train_results[k] += train_result[k]

        if self.step % self.hparams.log_interval_steps == 0:
            for k in self.train_results:
                v = self.train_results[k] / self.hparams.log_interval_steps 
                self.log.add_scalar( f'train/{k}', v, self.step) 

            self.train_results = defaultdict(float)
        
        if self.step % self.hparams.save_interval_steps == 0:
            self.save_checkpoint()


def check_args(args, hparams: Hyperparameter):
    if args.exp_dir is None:
        args.exp_dir = os.path.join('exps', datetime.datetime.now().strftime('exp-%Y%m%d-%H%M%S') )
    
        # 保存配置文件
        hparams.save_config( os.path.join( args.exp_dir, 'config.yaml' ) )

    # 是否需要进行数据预处理 
    if args.preprocess or not os.path.isfile( hparams.train_metadata_file ):
        if args.data_dir is None:
            raise RuntimeError('Must provide data directory for training.')
        from vocoder.preprocess import preprocess 
        preprocess(args.data_dir, hparams, args.temp_dir, args.device) 
    


def main():
    parser = argparse.ArgumentParser(
        description="Train LVC-WaveGAN (See detail in vocoder/train.py).")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--exp-dir", default=None, type=str,
                        help="the directory for saving expriment data, "
                        "including model checkpoints, log, results. ")
    parser.add_argument("--data-dir", default=None, type=str,
                        help="the directory containing .wav files for training")
    parser.add_argument("--temp-dir", default='temp', type=str,
                        help="the directory containing preprocess results")
    parser.add_argument("--restart", action="store_true", default=False,
                        help="Whether to restart a new training")
    parser.add_argument("--preprocess", action="store_true", default=False,
                        help="Whether force to preprocess data")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="checkpoint file path to load saving model")
    parser.add_argument("--device", default='cuda', type=str,
                        help="the device for training. (default: cuda:0)")
    args = parser.parse_args() 
    hparams = Hyperparameter( args.config ) 

    check_args( args, hparams ) 

    trainer = Trainer(args, hparams)

    try:
        trainer.train() 
    except KeyboardInterrupt:
        trainer.save_checkpoint() 
        trainer.log.flush()


if __name__ == "__main__":
    main() 