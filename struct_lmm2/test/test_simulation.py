import pytest
from numpy import logical_and, ones, zeros, sqrt
from numpy.linalg import matrix_rank
from numpy.random import RandomState
from numpy.testing import assert_, assert_allclose, assert_equal

from struct_lmm2._simulate import (
    column_normalize,
    create_environment_matrix,
    sample_covariance_matrix,
    sample_genotype,
    sample_maf,
    sample_persistent_effsizes,
    sample_gxe_effects,
    variances,
)


def test_sample_maf():
    random = RandomState(0)
    n_snps = 30
    maf_min = 0.2
    maf_max = 0.3
    mafs = sample_maf(n_snps, maf_min, maf_max, random)
    assert_(all(logical_and(maf_min <= mafs, mafs <= maf_max)))
    assert_(len(mafs) == n_snps)


def test_sample_genotype():
    random = RandomState(0)
    n_samples = 3
    n_snps = 30
    maf_min = 0.2
    maf_max = 0.3
    mafs = sample_maf(n_snps, maf_min, maf_max, random)
    G = sample_genotype(n_samples, mafs, random)
    assert_(G.shape == (n_samples, n_snps))

    A = set(list(G.ravel()))
    B = set([0.0, 1.0, 2.0])
    assert_(A - B == set())


def test_column_normalize():
    random = RandomState(0)
    n_samples = 10
    n_snps = 30
    maf_min = 0.2
    maf_max = 0.3
    mafs = sample_maf(n_snps, maf_min, maf_max, random)
    G = sample_genotype(n_samples, mafs, random)
    G = column_normalize(G)
    assert_allclose(G.mean(0), zeros(n_snps), atol=1e-7)
    assert_allclose(G.std(0), ones(n_snps))


def test_create_environment_matrix():
    n_samples = 5
    E = create_environment_matrix(n_samples)
    assert_equal(E, [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])


def test_sample_covariance_matrix():
    random = RandomState(0)
    n_samples = 5
    K = sample_covariance_matrix(n_samples, random)
    assert_(K.shape == (n_samples, n_samples))
    assert_allclose(K.diagonal().mean(), 1.0)
    assert_(matrix_rank(K) == K.shape[0])

    n_rep = 2
    K = sample_covariance_matrix(n_samples, random, n_rep)
    assert_(K.shape == (n_rep * n_samples, n_rep * n_samples))
    assert_allclose(K.diagonal().mean(), 1.0)
    assert_(matrix_rank(K) == K.shape[0])


def test_variances():
    r0 = 0.1
    v0 = 0.5

    v = variances(r0, v0)
    assert_allclose(sum(v.values()), 1.0)
    assert_allclose(v["v_e"], v["v_n"])
    assert_allclose(v["v_n"], v["v_k"])

    has_kinship = False
    v = variances(r0, v0, has_kinship)
    assert_allclose(sum(v.values()), 1.0)
    assert_allclose(v["v_e"], v["v_n"])


def test_sample_persistent_effsizes():
    random = RandomState(0)
    n_effects = 10
    causal_indices = [3, 5]
    variance = 0.9

    with pytest.raises(IndexError):
        sample_persistent_effsizes(2, causal_indices, variance, random)

    beta = sample_persistent_effsizes(n_effects, causal_indices, variance, random)
    assert_allclose(beta.mean(), 0.0, atol=1e-7)
    assert_allclose((beta ** 2).sum(), variance)


def test_sample_gxe_effects():
    random = RandomState(0)
    n_samples = 10
    n_snps = 30
    maf_min = 0.2
    maf_max = 0.3
    mafs = sample_maf(n_snps, maf_min, maf_max, random)
    G = sample_genotype(n_samples, mafs, random)
    G = column_normalize(G)
    E = create_environment_matrix(n_samples)
    E = column_normalize(E)
    E /= sqrt(E.shape[1])

    causal_indices = [3, 5, 8]
    variance = 0.9

    y2 = sample_gxe_effects(G, E, causal_indices, variance, random)
    assert_allclose(y2.mean(), 0.0, atol=1e-7)
    assert_allclose(y2.var(), variance)
