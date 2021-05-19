import inspect

from sacred import Ingredient

import functools

from exp.datasets.PySats import PySats
import sdsft
import numpy as np


ingredient = Ingredient('dataset')


@ingredient.config
def cfg():
    """Dataset configuration."""
    name = ''
    seed = np.random.randint(100000)
    bidder_function_type = ''
    parameters = {}
    

@ingredient.named_config
def GSVM():
    name = 'GSVM'
    bidder_function_type = 'single'
    

@ingredient.named_config
def MRVM():
    name = 'MRVM'
    bidder_function_type = 'sim'



@ingredient.capture
def get_instance(name, bidder_function_type, seed, parameters, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    
    sats = PySats.getInstance()
    
    if name == 'GSVM':
        world = sats.create_gsvm(seed, **parameters)
    elif name == 'MRVM':
         world = sats.create_mrvm(seed, **parameters)
    else:
        raise NotImplementedError()
    
    
    n = len(list(world.get_good_ids()))
    bidders = []
    for bidder_id in list(world.get_bidder_ids()):
        bidders += [sdsft.Bidder(world, bidder_id)]
    
    return bidders, n, seed
