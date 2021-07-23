import pytest
import scipy.stats as stats
from numpy import array, asarray, eye, hstack, ones, sqrt, stack, tile, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose

from struct_lmm2 import StructLMM2
from struct_lmm2._simulate import (
    column_normalize,
    create_environment_matrix,
    sample_covariance_matrix,
    sample_genotype,
    sample_maf,
    sample_persistent_effsizes,
    create_variances,
)


@pytest.fixture
def data():
    seed = 0
    random = RandomState(seed)
    n_samples = 100
    maf_min = 0.05
    maf_max = 0.45
    n_snps = 20
    # SNPs with persistent effect
    g_snps = [5, 6]
    # SNPs with interaction effect
    gxe_snps = [10, 11]
    # Contribution of interactions (proportion)
    r0 = 0.8
    v0 = 0.4

    # Simulate two environments
    E = create_environment_matrix(n_samples)

    mafs = sample_maf(n_snps, maf_min, maf_max, random)
    G = sample_genotype(n_samples, mafs, random)
    # We normalize the columns of G so that we have 𝔼[𝐠ᵀ𝐠] = 1.
    G = column_normalize(G)

    K = sample_covariance_matrix(n_samples, random)
    v = create_variances(r0, v0)
    beta_g = sample_effect_sizes(n_snps, g_snps, v["v_g"], random)
    beta_gxe = sample_effect_sizes(n_snps, gxe_snps, v["v_gxe"], random)
    y_g = G @ beta_g

    for i in range(n_snps):
        # beta_gxe = random.multivariate_normal(zeros(n_samples), sigma_gxe[i] * Sigma)
        beta_gxe = sigma_gxe[i] * u_gxe
        y_gxe += G[:, i] * beta_gxe

    # return {"y": y, "W": W, "E": E, "K": K, "G": G}


@pytest.mark.skip("Not working yet.")
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


@pytest.mark.skip("Not working yet.")
def test_structlmm2_interaction(data):
    y = data["y"]
    W = data["W"]
    E = data["E"]
    K = data["K"]
    G = data["G"]
    lmm2 = StructLMM2(y, W, E, K)
    print(lmm2.scan_interaction(G))
    pass
