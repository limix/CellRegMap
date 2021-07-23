from numpy import arange, asarray, eye, median, min, ones
from numpy.linalg import cholesky
from numpy.random import RandomState, default_rng
from numpy.testing import assert_, assert_allclose

from cellregmap import create_variances, sample_phenotype, sample_phenotype_gxe


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
    E = random.normal(size=[n_samples, n_env])

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
    E = random.normal(size=[n_samples, 2])

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
    E = random.normal(size=[n_samples, 2])

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
    E = random.normal(size=[n_samples, 2])

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


def test_struct_lmm2_inter_kinship_repetition():
    import numpy as np

    random = default_rng(20)

    n_individuals = 100
    # n_individuals = 200
    # n_individuals = 500
    # n_individuals = 250
    maf_min = 0.40
    # maf_min = 0.20
    # maf_min = 0.05
    maf_max = 0.45
    # n_snps = 250
    n_snps = 100
    # n_snps = 50
    # n_snps = 20
    # n_cells = 100
    # n_cells = 10
    # n_cells = 2
    n_cells = arange(n_individuals) + 1
    # n_cells = 1
    n_env = 2
    n_env_groups = 3
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]

    v = create_variances(r0, v0)

    s = sample_phenotype(
        offset=offset,
        n_individuals=n_individuals,
        n_snps=n_snps,
        n_cells=n_cells,
        n_env=n_env,
        n_env_groups=n_env_groups,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    slmm2 = StructLMM2(s.y, s.M, s.E, s.Lk)
    # idx = random.permutation(s.E.shape[0])
    # pvals = slmm2.scan_interaction(s.G, idx_E=idx)
    from scipy.stats.stats import pearsonr

    corr = {}
    corr_pvals = {}

    corr["y_e"] = [pearsonr(s.y_e, s.G[:, i])[0] for i in range(s.G.shape[1])]
    corr_pvals["y_e"] = [pearsonr(s.y_e, s.G[:, i])[1] for i in range(s.G.shape[1])]

    corr["y_n"] = [pearsonr(s.y_n, s.G[:, i])[0] for i in range(s.G.shape[1])]
    corr_pvals["y_n"] = [pearsonr(s.y_n, s.G[:, i])[1] for i in range(s.G.shape[1])]

    pvals, info = slmm2.scan_interaction(s.G)
    print(pearsonr(pvals, np.abs(corr["y_e"])))
    print(pearsonr(pvals, np.abs(corr["y_n"])))
    assert_(median(pvals) > 0.3)
    assert_(min(pvals) > 0.04)


def test_struct_lmm2_inter_kinship_repetition_gxe():
    import numpy as np

    random = default_rng(20)

    n_individuals = 100
    # n_individuals = 200
    # n_individuals = 500
    # n_individuals = 250
    maf_min = 0.40
    # maf_min = 0.20
    # maf_min = 0.05
    maf_max = 0.45
    # n_snps = 250
    n_snps = 100
    # n_snps = 50
    # n_snps = 20
    # n_cells = 100
    # n_cells = 10
    # n_cells = 2
    n_cells = arange(n_individuals) + 1
    # n_cells = 1
    n_env = 2
    n_env_groups = 3
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]

    v = create_variances(r0, v0)

    # Timing:
    # - n_individuals
    # - n_env_groups
    # - n_samples

    s = sample_phenotype_gxe(
        offset=offset,
        n_individuals=n_individuals,
        n_snps=n_snps,
        n_cells=n_cells,
        n_env=n_env,
        n_env_groups=n_env_groups,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    slmm2 = StructLMM2(s.y, s.M, s.E, s.Ls)
    # idx = random.permutation(s.E.shape[0])
    # pvals = slmm2.scan_interaction(s.G, idx_E=idx)
    from scipy.stats.stats import pearsonr

    corr = {}
    corr_pvals = {}

    corr["y_e"] = [pearsonr(s.y_e, s.G[:, i])[0] for i in range(s.G.shape[1])]
    corr_pvals["y_e"] = [pearsonr(s.y_e, s.G[:, i])[1] for i in range(s.G.shape[1])]

    corr["y_n"] = [pearsonr(s.y_n, s.G[:, i])[0] for i in range(s.G.shape[1])]
    corr_pvals["y_n"] = [pearsonr(s.y_n, s.G[:, i])[1] for i in range(s.G.shape[1])]

    pvals, info = slmm2.scan_interaction(s.G)
    print(pearsonr(pvals, np.abs(corr["y_e"])))
    print(pearsonr(pvals, np.abs(corr["y_n"])))
    assert_(median(pvals) > 0.3)
    assert_(min(pvals) > 0.04)


def test_struct_lmm2_inter_kinship_predict():
    random = RandomState(0)

    n_individuals = 100
    maf_min = 0.05
    maf_max = 0.45
    n_snps = 20
    n_cells = 2
    n_env_groups = 3
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]

    v = create_variances(r0, v0)

    s = sample_phenotype_gxe(
        offset=offset,
        n_individuals=n_individuals,
        n_snps=n_snps,
        n_cells=n_cells,
        n_env_groups=n_env_groups,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    slmm2 = StructLMM2(s.y, s.M, s.E, s.Ls)
    beta_g_s, beta_gxe_s = slmm2.predict_interaction(s.G, s.mafs)
    assert_allclose(beta_g_s[3], -0.07720025290188615)
    assert_allclose(beta_g_s[-1], 0.022415332334122483)
    assert_allclose(beta_gxe_s[1, 1], 0.010062608120425824)
    assert_allclose(beta_gxe_s[13, -1], -0.05566938579548831)


def test_struct_lmm2_estimate_aggregate_environment():
    random = default_rng(20)

    n_individuals = 100
    # n_individuals = 200
    # n_individuals = 500
    # n_individuals = 250
    maf_min = 0.40
    # maf_min = 0.20
    # maf_min = 0.05
    maf_max = 0.45
    # n_snps = 250
    n_snps = 100
    # n_snps = 50
    # n_snps = 20
    # n_cells = 100
    # n_cells = 10
    n_cells = 2
    # n_cells = arange(n_individuals) + 1
    # n_cells = 1
    n_env_groups = 3
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]

    v = create_variances(r0, v0)

    # Timing:
    # - n_individuals
    # - n_env_groups
    # - n_samples

    s = sample_phenotype_gxe(
        offset=offset,
        n_individuals=n_individuals,
        n_snps=n_snps,
        n_cells=n_cells,
        n_env_groups=n_env_groups,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    slmm2 = StructLMM2(s.y, s.M, s.E, s.Ls)
    import numpy as np
    for i in range(5):
        a = slmm2.estimate_aggregate_environment(s.G[:, i])
        print(np.var(a))
    b = slmm2.estimate_aggregate_environment(s.G[:, 10])
    c = slmm2.estimate_aggregate_environment(s.G[:, 11])
    print(np.var(b))
    print(np.var(c))
    pass
