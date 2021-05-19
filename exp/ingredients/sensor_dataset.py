import inspect

from sacred import Ingredient

import functools

import sdsft
import numpy as np


ingredient = Ingredient('dataset')

@ingredient.config
def cfg():
    """Dataset configuration."""
    name = ''
    set_function = None
    n = None

def information_gain(S, K, sigma=1):
    S = S.astype(np.bool)
    logdet = np.log(np.linalg.det(np.eye(S.sum()) + (1/sigma**2)*K[S][:, S]))
    return logdet

def load_information_gain(K, sigma=1):
    n = K.shape[0]
    sf = functools.partial(information_gain, K=K, sigma=sigma)
    s  = sdsft.WrapSetFunction(sf, False, True)
    return s, n

def load_berkeley(sigma=1):
    K = np.loadtxt('./exp/datasets/sensor_placement/Berkeley.csv', delimiter=',')
    return load_information_gain(K, sigma)

def load_rain(sigma=1):
    K = np.loadtxt('./exp/datasets/sensor_placement/Rain.csv', delimiter=',')
    return load_information_gain(K, sigma)

def load_california(sigma=1):
    K = np.loadtxt('./exp/datasets/sensor_placement/California.csv', delimiter=',')
    return load_information_gain(K, sigma)

@ingredient.named_config
def BERKELEY():
    name = 'berkeley_temperature'
    set_function, n = load_berkeley()

@ingredient.named_config
def CALIFORNIA():
    name = 'california_traffic'
    set_function, n = load_california()

@ingredient.named_config
def RAIN():
    name = 'rain'
    set_function, n = load_rain()
    

@ingredient.capture
def get_instance(name, n, set_function, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    s = set_function
    return s, n


