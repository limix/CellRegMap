import numpy as np
from glimix_core.lmm import LMM
from numpy import arange, asarray, clip, concatenate, inf, median, newaxis
from numpy.random import RandomState, default_rng
from numpy_sugar import epsilon
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from scipy.stats import chi2

from cellregmap import create_variances, sample_phenotype_fixed_gxe


def test_fixed_gxe():
    random = default_rng(20)

    n_individuals = 100
    # n_individuals = 200
    # n_individuals = 500
    # n_individuals = 250
    maf_min = 0.20
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
    n_env_groups = 2
    offset = 0.3
    r0 = 0.5
    v0 = 0.5
    g_causals = [5, 6]
    gxe_causals = [10, 11]

    v = create_variances(r0, v0)

    s = sample_phenotype_fixed_gxe(
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

    QS = economic_qs_linear(s.Lk)
    # QS = economic_qs(s.K)

    # Test 1:
    #   H0: s.y_e + s.y_k + s.y_n
    #   H1: s.y_g + s.y_e + s.y_k + s.y_n
    y = s.offset + s.y_g + 0*s.y_gxe + s.y_e + s.y_k + s.y_n
    M = concatenate([s.M, s.E], axis=1)
    lmm = LMM(y, M, QS, restricted=False)
    lmm.fit(verbose=False)
    scanner = lmm.get_fast_scanner()
    data1 = scanner.fast_scan(s.G)
    # Asserting to make sure that DoF==1
    assert data1["effsizes1"].ndim == 1
    dof = 1
    lml0 = scanner.null_lml()
    lml1 = data1["lml"]
    pv = test1_pvalues = lrt_pvalues(lml0, lml1, dof)
    breakpoint()
    print(pv[5], pv[6], pv[10], pv[11], median(pv[12:]))
    # arange     : 4.782253229179153e-192 1.0433583537506347e-196 0.01091492319670193 0.0010927629879084017 4.8750903985107926e-05
    # 2 n_cells  : 1.090975503699979e-08 4.899993877637627e-08 0.009198608906693148 0.4204345922908054 0.5335573850563654
    # 10 n_cells : 3.982864661957504e-08 1.9070517075766003e-08 0.09201571609949227 0.35721972899471366 0.543538454382785
    # 100 n_cells: 4.469523142654005e-08 3.0916502150797485e-09 0.07348502556111537 0.37843373021086246 0.5058335803047611

    # Test 2:
    #   H0: s.y_g + s.y_e + s.y_k + s.y_n
    #   H1: s.y_gxe + s.y_g + s.y_e + s.y_k + s.y_n
    y = s.offset + s.y_gxe + s.y_g + s.y_e + s.y_k + s.y_n
    assert min(abs(y - s.y)) < 1e-10
    data2 = {"lml0": [], "lml1": []}
    dof = 1
    for g in s.G.T:
        g = g[:, newaxis]
        M = concatenate([s.M, g, s.E], axis=1)
        lmm = LMM(y, M, QS, restricted=False)
        lmm.fit(verbose=False)
        scanner = lmm.get_fast_scanner()
        d = scanner.fast_scan(s.E * g)
        # Asserting to make sure that DoF==1
        assert d["effsizes1"].ndim == 1
        lml0 = scanner.null_lml()
        lml1 = d["lml"]
        data2["lml0"].append(lml0)
        data2["lml1"].append(lml1)

    lml0 = asarray(data2["lml0"])
    lml1 = concatenate(data2["lml1"])
    pv = test2_pvalues = lrt_pvalues(lml0, lml1, dof)
    breakpoint()
    print(pv[5], pv[6], pv[10], pv[11], median(pv[12:]))
    # arange     : 0.6084591769169483 3.6950357207986164e-10 3.457029241798351e-227 1.2283428436867437e-223 0.005415575098636543
    # 2 n_cells  : 0.07359659661565397 0.4350647685877501 2.3937927457201096e-07 1.9917895864290987e-16 0.43616571604445775
    # 10 n_cells : 0.3210415979502985 0.007255621562641298 1.109487704555233e-32 1.0877864951200334e-44 0.17000683356223117
    # 100 n_cells: 0.25813564725186755 8.150568151907782e-22 2.2250738585072014e-308 2.2250738585072014e-308 0.00013995763429074527
    pass

def compute_pvalues():
    stats["pv20"] = lrt_pvalues(stats["lml0"], stats["lml2"], stats["dof20"])

def lrt_pvalues(null_lml, alt_lmls, dof=1):
    """
    Compute p-values from likelihood ratios.

    These are likelihood ratio test p-values.

    Parameters
    ----------
    null_lml : float
        Log of the marginal likelihood under the null hypothesis.
    alt_lmls : array_like
        Log of the marginal likelihoods under the alternative hypotheses.
    dof : int
        Degrees of freedom.

    Returns
    -------
    pvalues : ndarray
        P-values.
    """
    lrs = clip(-2 * null_lml + 2 * asarray(alt_lmls, float), epsilon.super_tiny, inf)
    pv = chi2(df=dof).sf(lrs)
    return clip(pv, epsilon.super_tiny, 1 - epsilon.tiny)

