import sys

import numpy as np
from numpy import (
    asarray,
    concatenate,
    empty,
    eye,
    inf,
    newaxis,
    ones,
    sqrt,
    stack,
    trace,
    where,
    zeros,
)
from numpy.linalg import eigvalsh, inv, solve
from numpy.random import RandomState
from numpy_sugar.linalg import ddot, economic_qs, economic_svd
from scipy.linalg import sqrtm

from chiscore import davies_pvalue, optimal_davies_pvalue
from glimix_core.lmm import LMM
from struct_lmm import StructLMM


""" sample phenotype from the model:

    𝐲 = W𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + u + 𝛆

    𝛃 ∼ 𝓝(𝟎, b²Σ)
    𝐞 ∼ 𝓝(𝟎, e²Σ)
    𝛆 ∼ 𝓝(𝟎, 𝜀²I)
    Σ = EEᵀ
    u ~ 𝓝(𝟎, g²K)

"""


def _mod_liu(q, w):
    from chiscore import liu_sf

    (pv, dof_x, _, info) = liu_sf(q, w, [1] * len(w), [0] * len(w), True)
    return (pv, info["mu_q"], info["sigma_q"], dof_x)


def _qmin(pliumod):
    from numpy import zeros
    import scipy.stats as st

    # T statistic
    T = pliumod[:, 0].min()

    qmin = zeros(pliumod.shape[0])
    percentile = 1 - T
    for i in range(pliumod.shape[0]):
        q = st.chi2.ppf(percentile, pliumod[i, 3])
        mu_q = pliumod[i, 1]
        sigma_q = pliumod[i, 2]
        dof = pliumod[i, 3]
        qmin[i] = (q - dof) / (2 * dof) ** 0.5 * sigma_q + mu_q

    return qmin


# Let Σ = 𝙴𝙴ᵀ
# 𝐲 ∼ 𝓝(𝙼𝛂, 𝓋₀𝙳(ρ𝟏𝟏ᵀ + (1-ρ)Σ)𝙳 + 𝓋₁(aΣ + (1-a)𝙺) + 𝓋₂𝙸).

seed = int(sys.argv[1])
random = RandomState(seed)  # set a seed to replicate simulations
# set sample size
n_samples = 500
# simulate MAF (minor allele frequency) distribution
maf_min = 0.05
maf_max = 0.45
n_snps = 20

print(n_samples, "samples")
print(n_snps, "snps")
print(maf_min, "min MAF")
print(maf_max, "max MAF")

"simulate environments"

# two groups
group_size = n_samples // 2

E = zeros((n_samples, 2))

E[:group_size, 0] = 1
E[group_size:, 1] = 1

Sigma = E @ E.T

# import pdb; pdb.set_trace()

"simulate genotypes (for n_snps variants)"


# Simulate genotypes (for n_snps variants)
mafs = random.rand(n_snps) * (maf_max - maf_min) + maf_min

# simulate SNPs accordingly
G = []

for maf in mafs:
    g = random.choice(
        [0, 1, 2],
        p=[(1 - maf) ** 2, 1 - ((1 - maf) ** 2 + maf ** 2), maf ** 2],
        size=n_samples,
    )
    G.append(asarray(g, float))

# We normalize it such that the expectation of 𝔼[𝐠ᵀ𝐠] = 1.
# i.e. normalize columns

G = stack(G, axis=1)
G -= G.mean(0)
G /= G.std(0)

G0 = G.copy()
G0 /= sqrt(G0.shape[1])
K = G0 @ G0.T


"simulate two SNPs to have persistent effects and two to have interaction effects"
"one SNP in common, one unique to each category"

idxs_persistent = [5, 6]
idxs_gxe = [10, 11]

print("MAFs of causal SNPs")

print("{}\t{}".format(idxs_persistent[0], mafs[idxs_persistent[0]]))
print("{}\t{}".format(idxs_persistent[1], mafs[idxs_persistent[1]]))
print("{}\t{}".format(idxs_gxe[0], mafs[idxs_gxe[0]]))
print("{}\t{}".format(idxs_gxe[1], mafs[idxs_gxe[1]]))

# idxs_persistent = [5, 30]
# idxs_gxe = [30, 45]

# Variances
#
# 𝐲 ∼ 𝓝(1 + ∑ᵢ𝐠ᵢ𝛽_gᵢ, σ²_g⋅𝙳𝟏𝟏ᵀ𝙳 + σ²_gxe⋅𝙳Σ𝙳𝙳 + σ²_e⋅Σ + σ²_k⋅𝙺 + σ²_n⋅𝙸.
# σ²_g + σ²_gxe + σ²_e + σ²_k + σ²_n = 1
# σ²₁*ρ + σ²₁*(1-ρ) + σ²_e + σ²_k + σ²_n = 1

# The user will provide: σ²₁, ρ
# And we assume that σ²_e = σ²_k = σ²_n = v
# v = (1 - σ²₁*ρ + σ²₁*(1-ρ)) / 3
# σ²_e = a*σ²₂
# σ²_k = (1-a)*σ²₂


"simulate sigma parameters"

rho = 0.7  # contribution of interactions (proportion)
var_tot_g_gxe = 0.4

print(rho, "rho (prop var explained by GxE)")
print(var_tot_g_gxe, "tot variance G + GxE")

var_tot_g = (1 - rho) * var_tot_g_gxe
var_tot_gxe = rho * var_tot_g_gxe

var_g = var_tot_g / len(idxs_persistent)  # split effect across n signals
var_gxe = var_tot_gxe / len(idxs_gxe)

v = (1 - var_tot_gxe - var_tot_g) / 3
var_e = v  # environment effect only
var_k = v  # population structure effect ?
var_noise = v
# print(v)

""" (persistent) genotype portion of phenotype:

    𝐲_g = G 𝛃_g

    𝐲_g = ∑ᵢ𝐠ᵢ𝛽_gᵢ,

 where 𝐠ᵢ is the i-th column of 𝙶.

"""


# simulate (persistent) beta to have causal SNPs as defined
beta_g = zeros(n_snps)
beta_g[idxs_persistent] = random.choice([+1, -1], size=len(idxs_persistent))
beta_g /= beta_g.std()
beta_g *= var_g

"calculate genoytpe component of y"

y_g = G @ beta_g


""" GxE portion of phenotype:

     𝐲_gxe = ∑ᵢ gᵢ x 𝛃ᵢ

"""
# simulate (GxE) variance component to have causal SNPs as defined
sigma_gxe = zeros(n_snps)
sigma_gxe[idxs_gxe] = var_gxe

# for i in range(n_snps):
#     print('{}\t{}'.format(i,sigma_gxe[i]))

y_gxe = zeros(n_samples)
u_gxe = ones(n_samples)
u_gxe[group_size:] = -1

for i in range(n_snps):
    # beta_gxe = random.multivariate_normal(zeros(n_samples), sigma_gxe[i] * Sigma)
    beta_gxe = sigma_gxe[i] * u_gxe
    y_gxe += G[:, i] * beta_gxe


e = random.multivariate_normal(zeros(n_samples), v * Sigma)
u = random.multivariate_normal(zeros(n_samples), v * K)
eps = random.multivariate_normal(zeros(n_samples), v * eye(n_samples))

e0 = random.multivariate_normal(zeros(n_samples), v * 3 / 2 * Sigma)
eps0 = random.multivariate_normal(zeros(n_samples), v * 3 / 2 * eye(n_samples))


"sum all parts of y"
y = 1 + y_g + y_gxe + e + u + eps
y0 = 1 + y_g + y_gxe + e0 + eps0


p_values0 = []
p_values1 = []
p_values2 = []
p_values3 = []

print("testing using standard structLMM")

"test using struct LMM (standard)"


# y = y.reshape(y.shape[0], 1)

"Association test"

print(
    "p-values of association test SNPs",
    idxs_persistent,
    idxs_gxe,
    "should be causal (persistent + GxE)",
)

slmm = StructLMM(y0, M=np.ones(n_samples), E=E, W=E)
slmm.fit(verbose=False)

for i in range(n_snps):
    g = G[:, i]
    g = g.reshape(g.shape[0], 1)
    _p = slmm.score_2dof_assoc(g)
    print("{}\t{}".format(i, _p))
    p_values0.append(_p)

"Interaction test"

print("p-values of interaction test SNPs", idxs_gxe, "should be causal (GxE)")

for i in range(n_snps):
    g = G[:, i]
    # g = g.reshape(g.shape[0],1)
    M = np.ones(n_samples)
    M = np.stack([M, g], axis=1)
    slmm_int = StructLMM(y0, M=M, E=E, W=E)
    slmm_int.fit(verbose=False)
    _p = slmm_int.score_2dof_inter(g)
    print("{}\t{}".format(i, _p))
    p_values1.append(_p)


################################################
################################################
################################################
################################################

print("using structLMM 2 now")

"test using struct LMM 2 (in this case it should not be very different)"

# y = y.reshape(y.shape[0], 1)

Cov = {}
QS_a = {}
M = ones((n_samples, 1))

a_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
a_values = [1]

for a in a_values:
    Cov[a] = a * Sigma + (1 - a) * K
    QS_a[a] = economic_qs(Cov[a])

"Association test"

print(
    "p-values of association test SNPs",
    idxs_persistent,
    idxs_gxe,
    "should be causal (persistent + GxE)",
)

rhos = [0.0, 0.1 ** 2, 0.2 ** 2, 0.3 ** 2, 0.4 ** 2, 0.5 ** 2, 0.5, 0.999]

for i in range(n_snps):
    # print(i)
    g = G[:, i]
    g = g.reshape(g.shape[0], 1)
    best = {"lml": -inf, "a": 0, "v0": 0, "v1": 0, "beta": 0}
    for a in a_values:
        lmm = LMM(y0, E, QS_a[a], restricted=True)  # cov(y) = v0*(aΣ + (1-a)K) + v1*Is
        lmm.fit(verbose=False)
        if lmm.lml() > best["lml"]:
            best["lml"] = lmm.lml()
            best["a"] = a
            best["v0"] = lmm.v0
            best["v1"] = lmm.v1
            best["alpha"] = lmm.beta

    "H0 optimal parameters"
    alpha = lmm.beta[:-1]
    beta = lmm.beta[-1]
    # e²Σ + g²K = s²(aΣ + (1-a)K)
    # e² = s²*a
    # g² = s²*(1-a)
    s2 = lmm.v0  # s²
    eps2 = lmm.v1  # 𝜀²

    "H1 via score test"
    # Let K₀ = g²K + e²Σ + 𝜀²I
    # with optimal values e² and 𝜀² found above.
    K0 = lmm.covariance()
    X = concatenate((E, g), axis=1)

    # import pdb; pdb.set_trace()
    # Let P₀ = K⁻¹ - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹.
    K0iX = solve(K0, X)
    P0 = inv(K0) - K0iX @ solve(X.T @ K0iX, K0iX.T)

    # P₀𝐲 = K⁻¹𝐲 - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹𝐲.
    K0iy = solve(K0, y0)
    P0y = K0iy - solve(K0, X @ solve(X.T @ K0iX, X.T @ K0iy))

    # import pdb; pdb.set_trace()
    # The covariance matrix of H1 is K = K₀ + b²diag(𝐠)⋅Σ⋅diag(𝐠)
    # We have ∂K/∂b² = diag(𝐠)⋅Σ⋅diag(𝐠)
    # The score test statistics is given by
    # Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
    dK_G = ddot(g.ravel(), ddot(ones((n_samples, n_samples)), g.ravel()))
    dK_GxE = ddot(g.ravel(), ddot(Sigma, g.ravel()))
    sqrP0 = sqrtm(P0)
    Q_G = P0y.T @ dK_G @ P0y
    Q_GxE = P0y.T @ dK_GxE @ P0y

    # lambdas = zeros(len(rhos))
    lambdas = []
    Q = []
    for ii, rho in enumerate(rhos):
        # print(ii)
        Q.append((rho * Q_GxE + (1 - rho) * Q_G) / 2)
        dK = rho * dK_GxE + (1 - rho) * dK_G
        lambdas.append(eigvalsh((sqrP0 @ dK @ sqrP0) / 2))
        # lambdas[ii] = eigvalsh(sqrP0 @ dK @ sqrP0) / 2

    pliumod = stack([_mod_liu(Qi, lam) for Qi, lam in zip(Q, lambdas)], axis=0)
    qmin = _qmin(pliumod)

    # 3. Calculate quantites that occur in null distribution
    Px1 = P0 @ g
    m = 0.5 * (g.T @ Px1)
    goE = g * E
    PgoE = P0 @ goE
    ETxPxE = 0.5 * (goE.T @ PgoE)
    ETxPx1 = goE.T @ Px1
    ETxPx11xPxE = 0.25 / m * (ETxPx1 @ ETxPx1.T)
    ZTIminusMZ = ETxPxE - ETxPx11xPxE
    eigh = eigvalsh(ZTIminusMZ)

    eta = ETxPx11xPxE @ ZTIminusMZ
    vareta = 4 * trace(eta)

    OneZTZE = 0.5 * (g.T @ PgoE)
    tau_top = OneZTZE @ OneZTZE.T
    tau_rho = empty(len(rhos))

    for ii in range(len(rhos)):
        # print(ii)
        # tau_rho[ii] = rhos[ii] * m + (1 - rhos[ii]) / m * tau_top
        tau_rho[ii] = (1 - rhos[ii]) * m + (rhos[ii]) / m * tau_top


    MuQ = sum(eigh)
    VarQ = sum(eigh ** 2) * 2 + vareta
    KerQ = sum(eigh ** 4) / (sum(eigh ** 2) ** 2) * 12
    Df = 12 / KerQ

    # 4. Integration
    T = pliumod[:, 0].min()
    pvalue = optimal_davies_pvalue(
        qmin, MuQ, VarQ, KerQ, eigh, vareta, Df, tau_rho, rhos, T
    )

    # Final correction to make sure that the p-value returned is sensible
    multi = 3
    if len(rhos) < 3:
        multi = 2
    idx = where(pliumod[:, 0] > 0)[0]
    pval = pliumod[:, 0].min() * multi
    if pvalue <= 0 or len(idx) < len(rhos):
        pvalue = pval
    if pvalue == 0:
        if len(idx) > 0:
            pvalue = pliumod[:, 0][idx].min()

    print("{}\t{}".format(i, pvalue))
    p_values2.append(pvalue)
    # return pvalue

"Interaction test"


print("p-values of interaction test SNPs", idxs_gxe, "should be causal (GxE)")

for i in range(n_snps):
    g = G[:, i]
    g = g.reshape(g.shape[0], 1)
    Mg = concatenate((M, g), axis=1)
    best = {"lml": -inf, "a": 0, "v0": 0, "v1": 0, "beta": 0}
    for a in a_values:
        lmm = LMM(y0, Mg, QS_a[a], restricted=True)  # cov(y) = v0*(aΣ + (1-a)K) + v1*Is
        lmm.fit(verbose=False)
        if lmm.lml() > best["lml"]:
            best["lml"] = lmm.lml()
            best["a"] = a
            best["v0"] = lmm.v0
            best["v1"] = lmm.v1
            best["alpha"] = lmm.beta

    "H0 optimal parameters"
    alpha = lmm.beta[:-1]
    beta = lmm.beta[-1]
    # e²Σ + g²K = s²(aΣ + (1-a)K)
    # e² = s²*a
    # g² = s²*(1-a)
    s2 = lmm.v0  # s²
    eps2 = lmm.v1  # 𝜀²

    "H1 via score test"
    # Let K₀ = g²K + e²Σ + 𝜀²I
    # with optimal values e² and 𝜀² found above.
    K0 = lmm.covariance()
    X = concatenate((E, g), axis=1)

    # Let P₀ = K⁻¹ - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹.
    K0iX = solve(K0, X)
    P0 = inv(K0) - K0iX @ solve(X.T @ K0iX, K0iX.T)

    # P₀𝐲 = K⁻¹𝐲 - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹𝐲.
    K0iy = solve(K0, y0)
    P0y = K0iy - solve(K0, X @ solve(X.T @ K0iX, X.T @ K0iy))

    # The covariance matrix of H1 is K = K₀ + b²diag(𝐠)⋅Σ⋅diag(𝐠)
    # We have ∂K/∂b² = diag(𝐠)⋅Σ⋅diag(𝐠)
    # The score test statistics is given by
    # Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
    dK = ddot(g.ravel(), ddot(Sigma, g.ravel()))
    Q = (P0y.T @ dK @ P0y) / 2

    # Q is the score statistic for our interaction test and follows a linear combination
    # of chi-squared (df=1) distributions:
    # Q ∼ ∑λχ², where λᵢ are the non-zero eigenvalues of ½√P₀⋅∂K⋅√P₀.
    sqrP0 = sqrtm(P0)
    pval = davies_pvalue(Q, (sqrP0 @ dK @ sqrP0) / 2)
    print("{}\t{}".format(i, pval))
    p_values3.append(pval)
