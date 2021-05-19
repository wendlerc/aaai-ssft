import pickle
import numpy as np
from .common import SetFunction, SparseDSFT4Function


class Bidder(SetFunction):
    def __init__(self, world, bidder_id):
        self.world = world
        self.bidder_id = bidder_id
        self.call_counter = 0
        
    def __call__(self, indicators, count_flag=True):
        if not isinstance(indicators, np.ndarray):
            inds = indicators.toarray()
        else:
            inds = indicators
        if len(inds.shape) < 2:
            inds = inds[np.newaxis, :]
        if count_flag:
            self.call_counter += inds.shape[0]
        values = [self.world.calculate_value(self.bidder_id, ind) for ind in inds]
        return np.asarray(values)

    