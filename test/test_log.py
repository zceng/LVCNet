
import os, logging, time 
from vocoder.utils.log import Logger 

def test_logger():
    exp_dir = 'exps/exp-test' 
    os.makedirs( exp_dir, exist_ok=True )

    log = Logger( exp_dir )
    log.info('log test finish.') 

    while True:
        time.sleep(1)
        log.add_scalars('train', {'test': 0.2}, 1)

if __name__ == "__main__":
    test_logger() 


