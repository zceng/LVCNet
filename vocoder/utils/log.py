
from functools import wraps
import logging, os, time, sys 
from logging import DEBUG, INFO, WARN, ERROR

from torch.utils.tensorboard import SummaryWriter 

logging.basicConfig(
        stream=sys.stdout, 
        format='[ %(levelname)s ] %(message)s',
        level=DEBUG)

class Logger:

    def __init__(self, log_dir, level=DEBUG, tensorboard=True):
        os.makedirs( log_dir, exist_ok=True )

        self.logger = logging.getLogger('log') 
        self.logger.setLevel(level)
        handler = logging.FileHandler( 
            os.path.join( log_dir, time.strftime('%Y%m%d-%H%M%S.log') ), 
            mode='w',
            encoding='utf-8')
        handler.setFormatter( logging.Formatter('[ %(levelname)s, %(asctime)s ] %(message)s') ) 
        self.logger.addHandler( handler )
        self.handler = handler

        if tensorboard:
            self.tbwriter = SummaryWriter(log_dir) 

    def add_scalar(self, tag, value, step):
        self.tbwriter.add_scalar(tag, value, step) 

    def info(self, *msg, **kwargs):
        self.logger.info( *msg, **kwargs ) 

    def warn(self, *msg, **kwargs):
        self.logger.warn( *msg, **kwargs ) 

    def error(self, *msg, **kwargs):
        self.logger.error( *msg, **kwargs ) 

    def debug(self, *msg, **kwargs):
        self.logger.debug( *msg, **kwargs )

    def flush(self):
        self.tbwriter.flush() 
        self.handler.flush()



