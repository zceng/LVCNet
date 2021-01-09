
import argparse, glob, tqdm, os, random
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch

from .audio import TacotronSTFT, load_wav_to_torch 

from vocoder.datasets.utils import save_metadata
from vocoder.hparams import Hyperparameter 



def mel_transform(wav_files, mel_dir, mel_config, device, min_wav_length):
    # device = torch.device( device )
    # transfomer = MelSpectrogram( **mel_config ).to( device ) 
    taco_stft = TacotronSTFT( **mel_config )
    files = []
    with torch.no_grad():
        for fn in wav_files:
            audio, sr = load_wav_to_torch( fn, mel_config['sampling_rate'] ) 
            if audio.shape[1] < min_wav_length:
                print( 'skip {}, sr: {}, length: {}'.format(fn, sr, audio.shape[1]) )
                continue
            # audio = audio.to( device ) 
            mel, _ = taco_stft.mel_spectrogram( audio )
            mel_fn = os.path.join( mel_dir, os.path.basename(fn) + '.mel.npy' )
            np.save( mel_fn, mel[0].cpu().numpy().T )
            files.append( (fn, mel_fn) )
    return files



def preprocess( data_dir, 
                hparams: Hyperparameter, 
                temp_dir='temp', 
                device='cuda:0', 
                max_workers=4 ):
    '''Preprocess for LVC-WaveGAN.  
    Args: 
        data_dir (str): the directory containing .wav files.
        hparams (Hyperparameter): including parameter for calculating mel-spectrogram.
        temp_dir (str): the directory for saving preprocessing results.
        device (str): the cuda device for runing preprocessing.
        max_workers (int): the number of process worker.
    '''
    data_dir = os.path.abspath(data_dir)
    temp_dir = os.path.abspath(temp_dir)
    mel_dir = os.path.join( temp_dir, 'mels' ) 
    os.makedirs(mel_dir, exist_ok=True)
    mel_config = {
        'sampling_rate': hparams.sample_rate,
        'win_length': hparams.win_length,
        'hop_length': hparams.hop_length,
        'filter_length': hparams.n_fft,
        'mel_fmin': hparams.mel_fmin,
        'mel_fmax': hparams.mel_fmax,
        'n_mel_channels': hparams.n_mels,
    }
    min_wav_length = hparams.batch_mel_length * hparams.hop_length

    wav_files = glob.glob(f'{data_dir}/**/*.wav', recursive=True) 
    print('num of wavs:', len(wav_files))

    batch_size = 100 
    batch_num = int(np.ceil( len(wav_files) / batch_size ))
    batches = [ wav_files[ i*batch_size : (i+1)*batch_size ] for i in range( batch_num ) ]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [ executor.submit( mel_transform, batch, mel_dir, mel_config, 
                                     device, min_wav_length ) for batch in batches ] 
        for f in tqdm.tqdm( futures, desc='Preprocessing', total=batch_num ):
            results.extend( f.result() ) 

    save_metadata(results, os.path.join(temp_dir, 'metadata.txt'))

    # 产生训练、验证、测试训练集
    random.shuffle(results) 
    save_metadata(results[ : hparams.eval_sample_num ], hparams.eval_metadata_file )
    save_metadata(results[ -hparams.test_sample_num : ], hparams.test_metadata_file )
    save_metadata(results[ hparams.eval_sample_num : -hparams.test_sample_num ], 
                    hparams.train_metadata_file )



def main():
    parser = argparse.ArgumentParser(
        description="Preprocess for LVC-WaveGAN (See detail in vocoder/preprocess.py).")
    parser.add_argument("--data-dir", type=str, required=True, 
                        help="the directory containing .wav files")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--temp-dir", type=str, default='temp',
                        help="the directory to save preprocessing results")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="yaml format configuration file.")
    parser.add_argument("--device", default='cuda:0', type=str,
                        help="the device for training. (default: cuda:0)")
    args = parser.parse_args() 
    hparams = Hyperparameter( args.config ) 
    
    preprocess( args.data_dir, 
                hparams, 
                temp_dir=args.temp_dir, 
                device=args.device, 
                max_workers=args.max_workers)


if __name__ == '__main__':
    main()

