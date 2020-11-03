

class TrainStrategy:

    def __init__(self):
        pass 

    def train_step(self, batch, cur_step, model, loss, optimizer):
        return {'train_loss': 0}  

    def eval_step(self, batch, model, loss):
        return {'eval_loss': 0}

    def test_step(self, batch, model):
        return {'audio': 0}