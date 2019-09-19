import pytest
import scipy.stats as stats
from numpy import hstack, ones
from numpy.random import RandomState
from numpy.testing import assert_allclose
from struct_lmm import StructLMM

from struct_lmm2 import StructLMM2


@pytest.skip("Not working yet.")
def test_structlmm_int_2kinships():
    random = RandomState(0)
    n = 100
    c = 2
    y = random.randn(n)
    M = ones((n, 1))
    W = random.randn(n, c)
    E = random.randn(n, 4)

    p_values1 = []
    p_values2 = []

    for i in range(10):
        random = RandomState(i)
        g = random.randn(n)
        lmm2 = StructLMM2(y, W, E)
        lmm2.fit_null(g)
        _p = lmm2.score_2_dof(g)
        p_values2.append(_p)

        y = y.reshape(y.shape[0], 1)
        lmm = StructLMM(y, M, E)
        g = g.reshape(g.shape[0], 1)
        covs1 = hstack((W, g))
        lmm.fit(covs1)
        _p = lmm.score_2dof_inter(g)
        p_values1.append(_p)

    print(stats.pearsonr(p_values2, p_values1)[0])
    assert_allclose(p_values2, p_values1, rtol=1e-4)
