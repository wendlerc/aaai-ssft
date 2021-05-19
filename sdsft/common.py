from abc import ABC, abstractmethod
import numpy as np



class SetFunction(ABC):    
    def __call__(self, indicators, count_flag=True):
        """
            @param indicators: two dimensional np.array of type np.int32 or np.bool
            with one indicator vector per row
            @param count_flag: a flag indicating whether to count set function evaluations
            @returns: a np.array of set function evaluations
        """
        pass

class WrapSetFunction(SetFunction):
    def __init__(self, s, use_call_dict=False, use_loop=False):
        self.s = s
        self.call_counter = 0
        self.use_loop = use_loop
        if use_call_dict:
            self.call_dict = {}
        else:
            self.call_dict = None
        
        
    def __call__(self, indicator, count_flag=True):
        if len(indicator.shape) < 2:
            indicator = indicator[np.newaxis, :]
        
        result = []
        if self.call_dict is not None:
            for ind in indicator:
                key = tuple(ind.tolist())
                if key not in self.call_dict:
                    self.call_dict[key] = self.s(ind)
                    if count_flag:
                        self.call_counter += 1
                result += [self.call_dict[key]]
            return np.asarray(result)
        elif self.use_loop:
            result = []
            for ind in indicator:
                result += [self.s(ind)]
                if count_flag:
                    self.call_counter += 1
            return np.asarray(result)
        else:
            if count_flag:
                self.call_counter += indicator.shape[0]
            return self.s(indicator)


class SparseDSFT4Function(SetFunction):
    
    def __init__(self, frequencies, coefficients):
        """
            @param frequencies: two dimensional np.array of type np.int32 or np.bool 
            with one indicator vector per row
            @param coefficients: one dimensional np.array of corresponding Fourier 
            coeffients
        """
        self.freqs = frequencies
        self.coefs = coefficients
        self.call_counter = 0
        
        
    def __call__(self, indicators, count_flag=True):
        """
            @param indicators: two dimensional np.array of type np.int32 or np.bool
            with one indicator vector per row
            @param count_flag: a flag indicating whether to count set function evaluations
            @returns: a np.array of set function evaluations
        """
        ind = indicators
        freqs = self.freqs
        coefs = self.coefs
        if len(ind.shape) < 2:
            ind = ind[np.newaxis, :]
        active = freqs.dot(ind.T)
        active = active == 0
        res = (active * coefs[:, np.newaxis]).sum(axis=0)
        return res

    
class DSFT4OneHop(SetFunction):
    
    def __init__(self, n, weights, set_function):
        self.n = n
        self.weights = weights
        self.s = set_function
        self.call_counter = 0
    
    def __call__(self, indicators, count_flag=True, sample_optimal=True):
        if len(indicators.shape) < 2:
            indicators = indicators[np.newaxis, :]
        
        s = self.s
        weights = self.weights
        if sample_optimal:
            res = []
            for ind in indicators:
                nc = ind.shape[0]-np.sum(ind)
                if count_flag:
                    self.call_counter += (nc + 1)
                mask = ind.astype(np.int32)==0
                ind_shifted = np.tile(ind, [nc, 1])
                ind_shifted[:, mask] = np.eye(nc, dtype=ind.dtype)
                ind_one_hop = np.concatenate((ind[np.newaxis], ind_shifted), axis=0)
                weight_s0 = np.ones(1)*(1 + weights[True^mask].sum())
                active_weights = np.concatenate([weight_s0, weights[mask]])
                res += [(s(ind_one_hop)*active_weights).sum()]
            res = np.asarray(res)
        else:
            res = s(indicators)
            for i, weight in enumerate(weights):
                ind_shifted = indicators.copy()
                ind_shifted[:, i] = 1
                res += weight*s(ind_shifted)
            if count_flag:
                self.call_counter += (self.n + 1) * indicators.shape[0]
        return res
    
    def convertCoefs(self, estimate):
        freqs = estimate.freqs
        coefs = estimate.coefs
        coefs_new = []
        freqs = freqs.astype(np.bool)
        for key, value in zip(freqs, coefs):
            coefs_new += [value/(1 + self.weights[True^key].sum())]
        return SparseDSFT4Function(freqs.astype(np.int32), np.asarray(coefs_new))


def eval_sf(gt, estimate, n, n_samples=1000, err_type="rel", custom_samples=None, p=0.5):
    """
        @param gt: a SetFunction representing the ground truth
        @param estimate: a SetFunction 
        @param n: the size of the ground set
        @param n_samples: number of random measurements for the evaluation
        @param err_type: either mae or relative reconstruction error
    """
    if custom_samples is None:
        ind = np.random.binomial(1, p, (n_samples, n)).astype(np.bool)
    else:
        ind = custom_samples
    gt_vec = gt(ind, count_flag=False)
    est_vec = estimate(ind, count_flag=False)
    if err_type=="mae":
        return (np.linalg.norm(gt_vec - est_vec, 1)/n_samples)
    elif err_type=="rel":
        return np.linalg.norm(gt_vec - est_vec)/np.linalg.norm(gt_vec)
    elif err_type=="inf":
        return np.linalg.norm(gt_vec - est_vec, ord=np.inf)
    elif err_type=="res_quantiles":
        return np.quantile(np.abs(gt_vec - est_vec), [0.25, 0.5, 0.75])
    elif err_type=="quantiles":
        return np.quantile(np.abs(gt_vec), [0.25, 0.5, 0.75])
    elif err_type=="res":
        return gt_vec - est_vec
    elif err_type=="raw":
        return gt_vec, est_vec
    elif err_type=="R2":
        gt_mean = np.mean(gt_vec)
        return 1 - np.mean((est_vec - gt_vec)**2)/np.mean((gt_vec - gt_mean)**2)
    else:
        raise NotImplementedError("Supported error types: mae, rel, inf, res_quantiles, quantiles")


def gains(s, N, S0):
    max_value = -np.inf
    max_el = -1
    for element in N[True^S0]:
        curr_indicator = S0.copy()
        curr_indicator[element] = True
        curr_value = s(curr_indicator, count_flag=False)[0]
        if curr_value > max_value:
            max_value = curr_value
            max_el = element
        elif curr_value == max_value:
            if np.random.rand() > 0.5:
                max_value = curr_value
                max_el = element
    return max_el, max_value

def maximize_greedy(s, n, card, verbose=False, force_card=False):
    S0 = np.zeros(n, dtype=np.bool)
    N = np.arange(n)
    for t in range(card):
        i, value = gains(s, N, S0)
        if verbose:
            print('gains: i=%d, value=%.4f'%(i, value))
            print(S0.astype(np.int32))
        if value > 0 or force_card:
            S0[i] = 1
        else:
            break
    return S0, s(S0, count_flag=False)[0]
