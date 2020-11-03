
# LVCNet

LVCNet: Efficient Condition-Dependent Modeling Network for Waveform Generation


## Training and Test

### Parallel WaveGAN

0. prepare the data for training, evaluate, test.
    ```python
    python -m vocoder.preprocess --data-dir ../data/LJSpeech-1.1 --config configs/pwg.v1.yaml
    ```

1. Trainin Parallel WaveGAN
    ```python
    python -m vocoder.train --config configs/pwg.v1.yaml --exp-dir exps/exp.pwg.v1
    ```

2. Test Parallel WaveGAN 
    ```python 
    python -m vocoder.test --config configs/pwg.v1.yaml --exp-dir exps/exp.pwg.v1
    ```


### LVCNet WaveGAN

0. prepare the data for training, evaluate, test.
    ```python
    python -m vocoder.preprocess --data-dir ../data/LJSpeech-1.1 --config configs/lvcgan.v1.yaml
    ```

1. Trainin LVCNet WaveGAN
    ```python
    python -m vocoder.train --config configs/lvcgan.v1.yaml --exp-dir exps/exp.lvcgan.v1
    ```

2. Test LVCNet WaveGAN 
    ```python 
    python -m vocoder.test --config configs/lvcgan.v1.yaml --exp-dir exps/exp.lvcgan.v1
    ```


## Results 

### Tensorboard 

Use the tensorboard to view the experimental training process:

```
tensorboard --logdir exps
```


### Aduio Sample 

Audio Samples will be available soon.

