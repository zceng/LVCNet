
from .audio_mel import PWGAudioMelNoiseDataset, DataLoader

dataset_class_dict = {
    "PWGAudioMelNoiseDataset": PWGAudioMelNoiseDataset
}

def create_dataloader(dataset_classname, 
                      dataset_config,
                      batch_size=1,
                      collate_fn=None,
                      shuffle=False,
                      num_workers=0,
                      drop_last=False,
                      ) -> DataLoader:
    ''' create dataloader   
    Args: 
        dataset_classname (str) : the classname of dataset.
        dataset_config (dict): the config for dataset.
        ...
    Returns:
        Dataloader. 
    '''
    dataset = dataset_class_dict[ dataset_classname ]( **dataset_config )
    dataloader = DataLoader( dataset, 
                             batch_size=batch_size,
                             collate_fn=collate_fn,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             drop_last=drop_last)
    return dataloader  

