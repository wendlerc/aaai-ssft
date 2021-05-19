import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from sacred import Experiment
from tempfile import NamedTemporaryFile
import pandas as pd

import func_timeout as to

import sdsft

from exp.ingredients import model
from exp.ingredients import sensor_dataset as dataset

experiment = Experiment(
    'training',
    ingredients=[model.ingredient, dataset.ingredient]
)


@experiment.config
def cfg():
    n_samples = 10000 #number of samples for computing the error estimates
    timeout = 144*3600 # 144 h timeout
    card_max = 40 #maximal cardinality for the greedy optimization


@experiment.automain
def run(n_samples, card_max, timeout, _run, _log):

    result = {}
    # Get data
    s, n = dataset.get_instance()
    # Get model
    ft = model.get_instance(n)
    try: 
        start = time.time()
        estimate = to.func_timeout(timeout, ft.transform, args=[s])
        end = time.time()
        gt_vec, est_vec = sdsft.eval_sf(s, estimate, n, n_samples = n_samples, err_type='raw')
        rel = np.linalg.norm(gt_vec - est_vec)/np.linalg.norm(gt_vec)
        mae = np.mean(np.abs(gt_vec - est_vec))
        inf = np.linalg.norm(gt_vec - est_vec, ord=np.inf)        

        n_queries = s.call_counter
        t = end-start
        result['rel'] = result.get('rel', []) + [rel]
        result['mae'] = result.get('mae', []) + [mae]
        result['n_queries'] = result.get('n_queries', []) + [n_queries]
        result['time'] = result.get('time', []) + [t]
        result['freqs'] = result.get('freqs', []) + [estimate.freqs.tolist()]
        result['coefs'] = result.get('coefs', []) + [estimate.coefs.tolist()]
        print('mae %f, rel %f, n_q %d, t %f'%(mae, rel, n_queries, t), end='\r')
        _run.log_scalar('k', len(estimate.coefs))
        
        if card_max > 0:
            values_gt = []
            values_ft = []
            values_random = []
            for card in range(0, card_max+1):
                sensors, value = sdsft.maximize_greedy(s, n, card)
                sensors_ft, _ = sdsft.maximize_greedy(estimate, n, card)
                value_ft = s(sensors_ft)[0]
                values_gt += [value]
                values_ft += [value_ft]
                perm = np.random.permutation(n)
                ind = np.zeros(n, dtype=np.bool)
                ind[perm[:card]] = True
                values_random += [s(ind)[0]]

            with NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                plt.plot(values_gt, label='gt')
                plt.plot(values_ft, label='Fourier')
                plt.plot(values_random, label='random')
                plt.legend()
                plt.xlabel('cardinality constraint')
                plt.ylabel('information gain')
                plt.xlim(0, card_max)
                plt.ylim(bottom=0)
                plt.savefig(f.name, format='pdf')
                plt.close()
                _run.add_artifact(f.name, 'constrained_maximization.pdf')

            with NamedTemporaryFile(suffix='.csv', delete=False) as f:
                df = pd.DataFrame({"cards":np.arange(card_max+1), "gt": values_gt, 
                                   "fourier": values_ft, "random":values_random})
                df.to_csv(f.name, index=False, sep=',', decimal='.')
                _run.add_artifact(f.name, 'log_det.csv')

    except to.FunctionTimedOut:
        gt_vec, est_vec = 'timeout', 'timeout'
        t = 'timeout'
        rel = 'timeout'
        mae = 'timeout'
        n_queries = 'timeout'
        inf = 'timeout'

        result['rel'] = result.get('rel', []) + [rel]
        result['mae'] = result.get('mae', []) + [mae]
        result['n_queries'] = result.get('n_queries', []) + [n_queries]
        result['time'] = result.get('time', []) + [t]
        result['freqs'] = result.get('freqs', []) + ['timeout']
        result['coefs'] = result.get('coefs', []) + ['timeout']
        print('%d seconds timeout reached'%timeout, end='\r')
        
    
    
    _run.log_scalar('rel', rel)
    _run.log_scalar('mae', mae)
    _run.log_scalar('n_queries', n_queries)
    _run.log_scalar('time', t)
    _run.log_scalar('inf', inf)
      
    return result
