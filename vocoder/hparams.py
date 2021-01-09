

import yaml 


class Hyperparameter:
    ''' hyperparameter manager '''

    def __init__(self, config_file: str):
        ''' Hyperparameter  
        Args:
            config_file (str): the config file. 
        '''
        # Audio 
        self.sample_rate = 22050
        self.hop_length  = 256
        self.win_length  = 1024
        self.n_fft       = 1024
        self.n_mels      = 80
        self.mel_fmax   = 8000
        self.mel_fmin   = 70

        # Moel 
        self.model_name = 'ParallelWaveGAN'
        self.model_params = dict() 
        
        # Loss 
        self.loss_name = 'PWGLoss'
        self.loss_params = dict()

        # Optimizer
        self.opt_name = 'PWGOptimizer'
        self.opt_params = dict()

        # Strategy 
        self.strategy_name = "PWGStrategy"
        self.strategy_params = dict() 

        # Training
        self.dataset_classname   = 'PWGAudioMelNoiseDataset'
        self.dataset_num_workers = 5
        self.train_metadata_file = 'temp/metadata.train.txt' 
        self.batch_mel_length    = 52 
        self.train_batch_size    = 8
        self.max_train_steps     = 40000 
        self.log_interval_steps  = 100
        self.save_interval_steps = 10000

        # Evaluate
        self.eval_sample_num = 500 
        self.eval_metadata_file = 'temp/metadata.eval.txt'
        self.eval_interval_steps = 1000

        # Test 
        self.test_sample_num = 100 
        self.test_metadata_file = 'temp/metadata.test.txt' 


        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load( f )
            for k in config:
                self.__setattr__(k, config[k]) 

    def save_config(self, file):
        with open(file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.__dict__, f) 

    def __str__(self):
        return yaml.safe_dump(self.__dict__)


