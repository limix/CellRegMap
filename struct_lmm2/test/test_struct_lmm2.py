from struct_lmm import StructLMM
from struct_lmm2 import StructLMM2
from numpy.random import RandomState
from numpy.testing import assert_allclose
import numpy as np
import scipy.stats as stats

def test_structlmm_int_2kinships():

    random = RandomState(0)
    n = 100
    c = 2
    y = random.randn(n)
    W = random.randn(n, c)
    E = random.randn(n, 4)
    G = random.randn(n, 5)

    p_values1 = []
    p_values2 = []


    for i in range(10):
        random = RandomState(i)
        g = random.randn(n)
        X = np.concatenate((W, g[:, np.newaxis]), axis = 1)
        slmm_int = StructLMM2(y, W, E)
        null = slmm_int.fit_null(g)
        _p = slmm_int.score_2_dof(g)
        p_values2.append(_p)

        y = y.reshape(y.shape[0],1)
        slmm_int = StructLMM(y, E, W = E, rho_list = [0])
        g = g.reshape(g.shape[0],1)
        covs1 = np.hstack((W, g))
        null = slmm_int.fit_null(F = covs1, verbose = False)
        _p = slmm_int.score_2_dof(g)
        p_values1.append(_p)

    print(stats.pearsonr(p_values2, p_values1)[0])
    assert_allclose(p_values2, p_values1, rtol = 1e-4)
