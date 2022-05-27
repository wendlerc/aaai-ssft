import unittest
from sdsft import SparseSFT, SparseSFT3, WrapVector, ReverseSetFunction, SparseDSFT3Function, SparseDSFT4Function
import functools
import numpy as np
import sys
sys.path.append('..')


def int2indicator(A, n_groundset):
    indicator = [int(b) for b in bin(2**n_groundset + A)[3:][::-1]]
    indicator = np.asarray(indicator, dtype=np.int32)
    return indicator

class SSFTTest(unittest.TestCase):
    def test_ssft(self):
        M = np.asarray([[1., 1.], [1., 0.]])
        Finv = np.kron(np.kron(np.kron(np.kron(np.kron(M, M), M), M), M), M)
        n = 6
        
        inds = np.asarray([int2indicator(A, n) for A in range(2**n)])
        for _ in range(1000):
            coefs = 10*np.random.normal(0, 1, Finv.shape[0])
            perm = np.random.permutation(len(coefs))
            coefs[perm[5:]] = 0
            s_vec = Finv.dot(coefs)
            s = WrapVector(s_vec)
            ft = SparseSFT(n, flag_general=False, flag_print=False, eps=1e-6)
            s_est = ft.transform(s)
            s_est_vec = s_est(inds)
            if np.linalg.norm(s_vec - s_est_vec) >= 1e-6:
                print(s_est_vec)
                print(s_vec)
            self.assertTrue(np.linalg.norm(s_vec - s_est_vec) < 1e-6)
            
    def test_ssft3(self):
        M = np.asarray([[1., 0.], [1., 1.]])
        Finv = np.kron(np.kron(np.kron(np.kron(np.kron(M, M), M), M), M), M)
        n = 6
        
        inds = np.asarray([int2indicator(A, n) for A in range(2**n)])
        for _ in range(1000):
            coefs = 10*np.random.normal(0, 1, Finv.shape[0])
            perm = np.random.permutation(len(coefs))
            coefs[perm[5:]] = 0
            s_vec = Finv.dot(coefs)
            s = WrapVector(s_vec)
            ft = SparseSFT3(n, flag_general=False, flag_print=False, eps=1e-6)
            s_est = ft.transform(s)
            s_est_vec = s_est(inds)
            if np.linalg.norm(s_vec - s_est_vec) >= 1e-6:
                print(s_est_vec)
                print(s_vec)
            self.assertTrue(np.linalg.norm(s_vec - s_est_vec) < 1e-6)
            
    def test_ssft_degenerate(self):
        M = np.asarray([[1., 1.], [1., 0.]])
        Finv = np.kron(np.kron(np.kron(np.kron(np.kron(M, M), M), M), M), M)
        n = 6
        
        inds = np.asarray([int2indicator(A, n) for A in range(2**n)])
        s_hat = np.zeros(inds.shape[0])
        s_hat[1] = 1
        s_hat[3] = -1
        s_vec = Finv.dot(s_hat)
        s = WrapVector(s_vec)
        ft = SparseSFT(n, flag_general=False, flag_print=False, eps=1e-6)
        s_est = ft.transform(s)
        s_est_vec = s_est(inds)
        self.assertTrue(np.linalg.norm(s_vec - s_est_vec) > 1e-6)
        ft_plus = SparseSFT(n, flag_general=True, flag_print=False, eps=1e-6)
        s_est = ft_plus.transform(s)
        s_est_vec = s_est(inds)
        self.assertTrue(np.linalg.norm(s_vec - s_est_vec) < 1e-6)
        ft_full = SparseSFT(n, flag_general=False, flag_print=False, eps=0)
        s_est = ft_full.transform(s)
        s_est_vec = s_est(inds)
        self.assertTrue(np.linalg.norm(s_vec - s_est_vec < 1e-6))
        
    def test_ssft3_degenerate(self):
        M = np.asarray([[1., 0.], [1., 1.]])
        Finv = np.kron(np.kron(np.kron(np.kron(np.kron(M, M), M), M), M), M)
        n = 6
        
        inds = np.asarray([int2indicator(A, n) for A in range(2**n)])
        s_hat = np.zeros(inds.shape[0])
        s_hat[1] = 1
        s_hat[3] = -1
        s_vec = Finv.dot(s_hat)
        s = WrapVector(s_vec)
        ft = SparseSFT3(n, flag_general=False, flag_print=False, eps=1e-6)
        s_est = ft.transform(s)
        s_est_vec = s_est(inds)
        self.assertTrue(np.linalg.norm(s_vec - s_est_vec) > 1e-6)
        ft_plus = SparseSFT3(n, flag_general=True, flag_print=False, eps=1e-6)
        s_est = ft_plus.transform(s)
        s_est_vec = s_est(inds)
        self.assertTrue(np.linalg.norm(s_vec - s_est_vec) < 1e-6)
        ft_full = SparseSFT3(n, flag_general=False, flag_print=False, eps=0)
        s_est = ft_full.transform(s)
        s_est_vec = s_est(inds)
        self.assertTrue(np.linalg.norm(s_vec - s_est_vec < 1e-6))
        
    
    def test_ssft3_via_ssft(self):
        M = np.asarray([[1., 0.], [1., 1.]])
        Finv = np.kron(np.kron(np.kron(np.kron(np.kron(M, M), M), M), M), M)
        n = 6
        
        inds = np.asarray([int2indicator(A, n) for A in range(2**n)])
        for _ in range(1000):
            coefs = 10*np.random.normal(0, 1, Finv.shape[0])
            perm = np.random.permutation(len(coefs))
            coefs[perm[5:]] = 0
            s_vec = Finv.dot(coefs)
            s = WrapVector(s_vec)
            r = ReverseSetFunction(s)
            ft = SparseSFT(n, flag_general=False, flag_print=False, eps=1e-6)
            r_est = ft.transform(r)
            s_est = SparseDSFT3Function(r_est.freqs, r_est.coefs)
            s_est_vec = s_est(inds)
            if np.linalg.norm(s_vec - s_est_vec) >= 1e-6:
                print(r_est.freqs)
                print(r_est.coefs)
                print(inds[coefs!=0])
                print(coefs)

            self.assertTrue(np.linalg.norm(s_vec - s_est_vec) < 1e-6)
        
        
    def test_ssft_via_ssft3(self):
        M = np.asarray([[1., 1.], [1., 0.]])
        Finv = np.kron(np.kron(np.kron(np.kron(np.kron(M, M), M), M), M), M)
        n = 6
        
        inds = np.asarray([int2indicator(A, n) for A in range(2**n)])
        for _ in range(1000):
            coefs = 10*np.random.normal(0, 1, Finv.shape[0])
            perm = np.random.permutation(len(coefs))
            coefs[perm[5:]] = 0
            s_vec = Finv.dot(coefs)
            s = WrapVector(s_vec)
            r = ReverseSetFunction(s)
            ft = SparseSFT3(n, flag_general=False, flag_print=False, eps=1e-6)
            r_est = ft.transform(r)
            s_est = SparseDSFT4Function(r_est.freqs, r_est.coefs)
            s_est_vec = s_est(inds)
            if np.linalg.norm(s_vec - s_est_vec) >= 1e-6:
                print(r_est.freqs)
                print(r_est.coefs)
                print(inds[coefs!=0])
                print(coefs)

            self.assertTrue(np.linalg.norm(s_vec - s_est_vec) < 1e-6)
          
    
if __name__ == '__main__':
    unittest.main()
