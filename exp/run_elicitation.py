import os

import numpy as np
import matplotlib.pyplot as plt
import time
from sacred import Experiment
from tempfile import NamedTemporaryFile

import sdsft

from exp.ingredients import model
from exp.ingredients import auction_dataset as dataset

experiment = Experiment(
    'training',
    ingredients=[model.ingredient, dataset.ingredient]
)


@experiment.config
def cfg():
    repetitions = 10
    n_samples = 10000


@experiment.automain
def run(repetitions, n_samples, _run, _log):

    result = {}
    for rep in range(repetitions):
        # Get data
        seed = np.random.randint(100000)
        bidders, n, seed = dataset.get_instance(seed=seed)
        # Get model
        ft = model.get_instance(n)
        result['seed'] = result.get('seed', []) + [seed]
        for idx, bidder in enumerate(bidders):
            start = time.time()
            estimate = ft.transform(bidder)
            end = time.time()
            n_queries = bidder.call_counter
            rel = sdsft.eval_sf(bidder, estimate, n, n_samples = n_samples)
            mae = sdsft.eval_sf(bidder, estimate, n, n_samples = n_samples, 
                                err_type='mae')
            
            t = end-start
            
            bidder_result = result.get('Bidder %d'%idx, {})
            bidder_result['rel'] = bidder_result.get('rel', []) + [rel]
            bidder_result['mae'] =bidder_result.get('mae', []) + [mae]
            bidder_result['n_queries'] = bidder_result.get('n_queries', []) + [n_queries]
            bidder_result['time'] = bidder_result.get('time', []) + [t]
            bidder_result['freqs'] = bidder_result.get('freqs', []) + [estimate.freqs.tolist()]
            bidder_result['coefs'] = bidder_result.get('coefs', []) + [estimate.coefs.tolist()]
            result['Bidder %d'%idx] = bidder_result
            
            _run.log_scalar('Bidder %d, rel'%idx, rel, rep)
            _run.log_scalar('Bidder %d, mae'%idx, mae, rep)
            _run.log_scalar('Bidder %d, n_queries'%idx, n_queries, rep)
            _run.log_scalar('Bidder %d, time'%idx, t, rep)
            _run.log_scalar('Bidder %d, k'%idx, len(estimate.coefs), rep)
            print()
            print('%.2f'%(100*(rep * len(bidders) + idx+1)/(repetitions*len(bidders))), '%',
                  'rep %d, bidder %d, mae %f, rel %f, n_q %d, t %f'%(rep, idx, mae, rel, n_queries, t), end='\r')
            

    return result
