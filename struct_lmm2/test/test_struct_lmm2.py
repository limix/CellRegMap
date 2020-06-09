from numpy.linalg import cholesky
from numpy import ones, asarray, eye, median, min
from numpy.testing import assert_, assert_allclose
from struct_lmm2 import StructLMM2
from numpy.random import RandomState
from struct_lmm2 import sample_phenotype, create_variances
from numpy.random import RandomState


# from struct_lmm import StructLMM
# @pytest.mark.skip
def test_struct_lmm2_assoc():
    random = RandomState(0)

    n_samples = 50
    maf_min = 0.05
    maf_max = 0.45
    n_snps = 20
    n_rep = 1
    n_env = 2
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]

    v = create_variances(r0, v0)
    E = random.randn(n_samples, n_env)

    s = sample_phenotype(
        offset=offset,
        E=E,
        n_samples=n_samples,
        n_snps=n_snps,
        n_rep=n_rep,
        n_env=n_env,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    M = ones((n_samples, 1))
    # slmm = StructLMM(s.y.copy(), M=M, E=s.E, W=s.E)
    # slmm.fit(verbose=False)

    # p_values0 = []
    # print()
    # for i in range(n_snps):
    #     g = s.G[:, [i]]
    #     p = slmm.score_2dof_assoc(g)
    #     print("{}\t{}".format(i, p))
    #     p_values0.append(p)

    slmm2 = StructLMM2(s.y, M, s.E)
    slmm2.fit_null_association()
    p_values1 = slmm2.scan_association(s.G)
    # for i, pv in enumerate(p_values1):
    #     print("{}\t{}".format(i, pv))


def test_struct_lmm2_inter():
    random = RandomState(3)

    n_samples = 280
    # n_samples = 500
    maf_min = 0.05
    maf_max = 0.45
    n_snps = 20
    n_rep = 1
    n_env = 4
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]
    E = random.randn(n_samples, 2)

    v = create_variances(r0, v0)

    s = sample_phenotype(
        offset=offset,
        E=E,
        n_samples=n_samples,
        n_snps=n_snps,
        n_rep=n_rep,
        n_env=n_env,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    M = ones((n_samples, 1))

    # p_values0 = []
    # for i in range(n_snps):
    #     g = s.G[:, [i]]
    #     Mg = concatenate([M, g], axis=1)
    #     slmm_int = StructLMM(s.y.copy(), M=Mg, E=s.E, W=s.E)
    #     slmm_int.fit(verbose=False)
    #     p = slmm_int.score_2dof_inter(g)
    #     print("{}\t{}".format(i, p))
    #     p_values0.append(p)

    # TODO: add kinship to test it properly
    slmm2 = StructLMM2(s.y, M, s.E)
    pvals = slmm2.scan_interaction(s.G)

    causal_pvalues = [pvals[i] for i in range(len(pvals)) if i in gxe_causals]
    noncausal_pvalues = [pvals[i] for i in range(len(pvals)) if i not in gxe_causals]
    causal_pvalues = asarray(causal_pvalues)
    noncausal_pvalues = asarray(noncausal_pvalues)

    assert_(all(causal_pvalues < 1e-7))
    assert_(all(noncausal_pvalues > 1e-3))


def test_struct_lmm2_inter_kinship():
    random = RandomState(8)

    n_samples = 346
    maf_min = 0.05
    maf_max = 0.45
    n_snps = 20
    n_rep = 1
    n_env = 3
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]
    E = random.randn(n_samples, 2)

    v = create_variances(r0, v0)

    s = sample_phenotype(
        offset=offset,
        E=E,
        n_samples=n_samples,
        n_snps=n_snps,
        n_rep=n_rep,
        n_env=n_env,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    M = ones((n_samples, 1))

    G_kinship = cholesky(s.K + eye(n_samples) * 1e-7)
    slmm2 = StructLMM2(s.y, M, s.E, G_kinship)
    pvals = slmm2.scan_interaction(s.G)

    causal_pvalues = [pvals[i] for i in range(len(pvals)) if i in gxe_causals]
    noncausal_pvalues = [pvals[i] for i in range(len(pvals)) if i not in gxe_causals]
    causal_pvalues = asarray(causal_pvalues)
    noncausal_pvalues = asarray(noncausal_pvalues)

    assert_(all(causal_pvalues < 1e-7))
    assert_(all(noncausal_pvalues > 1e-3))


def test_struct_lmm2_inter_kinship_permute():
    random = RandomState(2)

    n_samples = 500
    maf_min = 0.05
    maf_max = 0.45
    n_snps = 20
    n_rep = 1
    n_env = 3
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]
    E = random.randn(n_samples, 2)

    v = create_variances(r0, v0)

    s = sample_phenotype(
        offset=offset,
        E=E,
        n_samples=n_samples,
        n_snps=n_snps,
        n_rep=n_rep,
        n_env=n_env,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    M = ones((n_samples, 1))

    G_kinship = cholesky(s.K + eye(n_samples) * 1e-7)
    slmm2 = StructLMM2(s.y, M, s.E, G_kinship)
    random = RandomState(1)
    idx = random.permutation(s.E.shape[0])
    pvals = slmm2.scan_interaction(s.G, idx_E=idx)
    assert_(median(pvals) > 0.3)
    assert_(min(pvals) > 0.04)


def test_struct_lmm2_inter_kinship_predict():
    random = RandomState(0)

    n_samples = 100
    maf_min = 0.05
    maf_max = 0.45
    n_snps = 20
    n_rep = 1
    n_env = 3
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]
    E = random.randn(n_samples, 2)

    v = create_variances(r0, v0)

    s = sample_phenotype(
        offset=offset,
        E=E,
        n_samples=n_samples,
        n_snps=n_snps,
        n_rep=n_rep,
        n_env=n_env,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    M = ones((n_samples, 1))

    G_kinship = cholesky(s.K + eye(n_samples) * 1e-7)
    E = random.randn(n_samples, 2)
    slmm2 = StructLMM2(s.y, M, E, G_kinship)
    beta_stars = slmm2.predict_interaction(s.G)
    assert_allclose(beta_stars.mean(), 0.030532771936166866)
