
import torch
import numpy as np
from scipy.signal import get_window
import librosa


def window_sumsquare(window, n_frames, hop_length, win_length,
                     n_fft, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa.util.normalize(win_sq, norm=norm)**2
    win_sq = librosa.util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)
          ] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    x = 20 * torch.log10(torch.clamp(x, min=clip_val)) - 20
    x = torch.clamp((x + 100) / 100, 0.0, 1.0) 
    return x


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    x = x * 100 - 100 
    x = torch.pow(10, x/20 + 1)
    return x


def load_wav_to_torch(wav_file, sample_rate):
    ''' load wav and convert into Tensor.  

    Args:
        wav_file (str): the path of wav file.  
        sample_rate (int): sample_rate 

    Returns: 
        audio (Tensor): shape (1, L) 
        sample_rate (int) 
    '''
    data, sr = librosa.load(wav_file, sample_rate)
    if len(data.shape) != 1:
        raise ValueError( f"the audio ({wav_file}) is not single channel." ) 
    return torch.FloatTensor(data).unsqueeze(0), sr

