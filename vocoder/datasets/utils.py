



def read_metadata(metadata_path, split='|'):
    ''' read data from metadata file. 

    Args:
        metadata_path (str): the path of metadata file. 
        split (str): the char to split each line in metadata file. 
            default: '|'  
    Returns:
        list: data from metadata file. 
    '''
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = f.readlines() 
    data = [ d.strip().split('|') for d in data ] 
    return data 


def save_metadata(data, metadata_path, split='|'):
    '''save data to file as metadata. 

    Args:
        data (list): data
        metadata_path (str): path for saving file.
        split (str): the char to join each element of data.  
    Returns:
        None
    '''
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for d in data: 
            line = split.join( d ) 
            line.replace('\n',' ')
            f.write(line + '\n')