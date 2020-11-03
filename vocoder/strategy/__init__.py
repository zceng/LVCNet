


from .base import TrainStrategy
from .pwg_strategy import PWGStrategy

strategy_classes = {
    "PWGStrategy": PWGStrategy
}


def create_strategy(name, params) -> TrainStrategy:
    return strategy_classes[ name ](**params)