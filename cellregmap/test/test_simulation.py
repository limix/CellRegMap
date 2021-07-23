import pytest
from numpy import logical_and, ones, tile, zeros
from numpy.linalg import matrix_rank
from numpy.random import RandomState
from numpy.testing import assert_, assert_allclose, assert_equal

from cellregmap._simulate import (
    column_normalize,
    create_environment_matrix,
    create_variances,
    sample_covariance_matrix,
    sample_genotype,
    sample_gxe_effects,
    sample_maf,
    sample_noise_effects,
    sample_persistent_effsizes,
    sample_phenotype,
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

    v = create_variances(r0, v0)
    assert_allclose(v.g + v.gxe + v.k + v.e + v.n, 1.0)
    assert_allclose(v.e, v.n)
    assert_allclose(v.n, v.k)

    has_kinship = False
    v = create_variances(r0, v0, has_kinship)
    assert_allclose(v.g + v.gxe + v.e + v.n, 1.0)
    assert_allclose(v.e, v.n)
    assert_equal(v.k, None)


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
    n_rep = 2
    maf_min = 0.2
    maf_max = 0.3
    mafs = sample_maf(n_snps, maf_min, maf_max, random)

    G = sample_genotype(n_samples, mafs, random)
    G = tile(G, (n_rep, 1))
    G = column_normalize(G)

    E = random.randn(5, 5)
    E = create_environment_matrix(E, n_samples, n_rep, 3, random)
    E = column_normalize(E)

    causal_indices = [3, 5, 8]
    variance = 0.9

    y2 = sample_gxe_effects(G, E, causal_indices, variance, random)
    assert_allclose(y2.mean(), 0.0, atol=1e-7)
    assert_allclose(y2.var(), variance)


# TODO: put it back
# def test_sample_environment_effects():
#     random = RandomState(0)
#     n_samples = 10

#     E = random.randn(5, 5)
#     E = create_environment_matrix(E, n_samples, 2, 3, random)
#     E = column_normalize(E)

#     variance = 0.3
#     y3 = sample_environment_effects(E, variance, random)

#     assert_allclose(y3.mean(), 0.0, atol=1e-7)
#     assert_allclose(y3.var(), variance)


# def test_sample_population_effects():
#     random = RandomState(0)
#     n_samples = 10
#     variance = 0.4
#     n_rep = 1
#     K = sample_covariance_matrix(n_samples, random, n_rep)
#     y4 = sample_population_effects(K, variance, random)

#     assert_allclose(y4.mean(), 0.0, atol=1e-7)
#     assert_allclose(y4.var(), variance)


def test_sample_noise_effects():
    random = RandomState(0)
    n_samples = 10

    variance = 0.3
    y5 = sample_noise_effects(n_samples, variance, random)

    assert_allclose(y5.mean(), 0.0, atol=1e-7)
    assert_allclose(y5.var(), variance)


def test_sample_phenotype():
    from numpy import corrcoef

    random = RandomState(0)

    r0 = 0.1
    v0 = 0.5
    v = create_variances(r0, v0)

    E = random.randn(30, 10)

    offset = 0.3
    n_samples = 500
    n_rep = 2
    n_env = 3
    s = sample_phenotype(
        offset=offset,
        E=E,
        n_samples=n_samples,
        n_snps=300,
        n_rep=n_rep,
        n_env=n_env,
        maf_min=0.1,
        maf_max=0.4,
        g_causals=[3, 4],
        gxe_causals=[4, 5],
        variances=v,
        random=random,
    )
    assert_allclose(
        s.y_g.var() + s.y_gxe.var() + s.y_n.var() + s.y_e.var() + s.y_k.var(), 1.0
    )
    assert_(abs(corrcoef(s.y_g, s.y_gxe)[0, 1]) < 0.1)
    assert_(abs(corrcoef(s.y_g, s.y_n)[0, 1]) < 0.1)
    assert_(abs(corrcoef(s.y_g, s.y_e)[0, 1]) < 0.1)
    assert_(abs(corrcoef(s.y_g, s.y_k)[0, 1]) < 0.1)

    assert_(abs(corrcoef(s.y_gxe, s.y_n)[0, 1]) < 0.1)
    assert_(abs(corrcoef(s.y_gxe, s.y_e)[0, 1]) < 0.1)
    assert_(abs(corrcoef(s.y_gxe, s.y_k)[0, 1]) < 0.1)

    assert_(abs(corrcoef(s.y_n, s.y_e)[0, 1]) < 0.1)
    assert_(abs(corrcoef(s.y_n, s.y_k)[0, 1]) < 0.1)

    assert_(abs(corrcoef(s.y_e, s.y_k)[0, 1]) < 0.1)
    assert_allclose(s.y, offset + s.y_g + s.y_gxe + s.y_n + s.y_e + s.y_k)
    assert_allclose(s.y.mean(), offset)

    assert_(s.E.shape[0] == n_samples * n_rep)
    assert_(s.E.shape[1] == n_env)
