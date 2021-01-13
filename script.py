from numpy_sugar.linalg import economic_qs, economic_qs_linear
from numpy.random import RandomState, default_rng
from numpy import arange, concatenate, newaxis, median
from glimix_core.lmm import LMM
from scipy.stats import chi2
from numpy_sugar import epsilon
from numpy import asarray, clip, inf, ones, sqrt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from struct_lmm2 import StructLMM2, create_variances, sample_phenotype_gxe

def _symmetric_decomp(H):
    # H = L @ L.T
    # Returns L
    n = min((2,) + H.shape)
    last_expl_var = inf
    while last_expl_var > epsilon.tiny:
        pca = PCA(n_components=n).fit(H)
        if n == min(H.shape):
            break

        last_expl_var = pca.explained_variance_[-1]
        n = min((n * 2,) + H.shape)

    L = pca.components_.T * sqrt(pca.singular_values_)
    return L


random = default_rng(20)
n_individuals = 100

maf_min = 0.20
maf_max = 0.45

# n_snps = 30
n_snps = 100
# n_snps = 500

# n_cells = 100
# n_cells = 10
# n_cells = 2
n_cells = arange(n_individuals) + 1

n_env_groups = 2
offset = 0.3

# indices of causal SNPs
g_causals = [5, 6]
gxe_causals = [10, 11]

# weight of genetic variance explained by GxE
r0 = 0.5
# r0 = 0
# r0 = 1

# total variance explained by genetics (G + GxE)
v0 = 0.5
# v0 = 0


v = create_variances(r0, v0)

s = sample_phenotype_gxe(
        offset=offset,
        n_individuals=n_individuals,
        n_snps=n_snps,
        n_cells=n_cells,
        n_env_groups=n_env_groups,
        n_env = 1,
        maf_min=maf_min,
        maf_max=maf_max,
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )


#K_E = s.K * (s.E @ s.E.T)
#hKE = _symmetric_decomp(K_E)

M = ones((s.K.shape[0], 1))
slmm2 = StructLMM2(s.y, M, s.E, s.Lk)

pv = slmm2.scan_interaction(s.G)
p = pv[0]

breakpoint()
print(p[5], p[6], p[10], p[11], median(p[12:]))
