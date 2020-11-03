
import tqdm 
from vocoder.datasets import create_dataloader 

def test_mel_audio_dataset():
    dataset_config = {
        'metadata_file': 'temp/metadata.txt',
        'hop_length': 256,
        'batch_mel_length': 64
    }

    dataloader = create_dataloader( "AudioMelNoiseDataset", 
                                    dataset_config=dataset_config,
                                    batch_size=4,
                                    shuffle=True,
                                    num_workers=4,
                                    drop_last=False ) 
    for batch in tqdm.tqdm(dataloader):
        wavs, mels, noises = batch 
        


if __name__ == "__main__":
    test_mel_audio_dataset()
