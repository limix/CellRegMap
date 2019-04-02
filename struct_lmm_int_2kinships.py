"""
Mixed-model with genetic effect heterogeneity.

The extended StructLMM model is ::

    𝐲 = W𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + u + 𝛆,

where ::

    𝐠⊙𝛃 = ∑ᵢ𝐠ᵢ𝛽ᵢ
    𝛃 ∼ 𝓝(𝟎, b²Σ)
    𝐞 ∼ 𝓝(𝟎, e²Σ)
    𝛆 ∼ 𝓝(𝟎, 𝜀²I)
    Σ = EEᵀ
    u ~ 𝓝(𝟎, g²K)

If one considers 𝛽 ∼ 𝓝(0, p²), we can insert
𝛽 into 𝛃 ::

    𝛃_ ∼ 𝓝(𝟎, p²𝟏𝟏ᵀ + b²Σ)

"""
from numpy import concatenate, inf, newaxis
from numpy.linalg import eigvalsh, inv, solve
from numpy.random import RandomState
from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, economic_qs
from scipy.linalg import sqrtm

from glimix_core.lmm import LMM

random = RandomState(0)
n = 30
c = 2

y = random.randn(n)
W = random.randn(n, c)
g = random.randn(n)
E = random.randn(n, 4)
Sigma = E @ E.T

# 𝐲 = W𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + 𝛆,
# 𝐞 ∼ 𝓝(𝟎, e²Σ)
# Σ = EEᵀ
QS = economic_qs(Sigma)

# 𝐲 = W𝛂 + 𝐞 + 𝛆
# 𝓝(𝐲 | W𝛂, e²Σ + 𝜀²I)

"""
Interaction test
----------------
H0: b² = 0 => 𝐲 = W𝛂 + 𝐠𝛽 + 𝐞 + u + 𝛆
    𝐲 ∼ 𝓝(W𝛂 + 𝐠𝛽, e²Σ + g²K + 𝜀²I)
H1: b² > 0 => 𝐲 = W𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + u + 𝛆
    𝐲 ∼ 𝓝(W𝛂 + 𝐠𝛽, e²Σ + g²K + 𝜀²I + b²Σ)
"""
X = concatenate((W, g[:, newaxis]), axis=1)
lmm = LMM(y, X, QS)
lmm.fit(verbose=False)

best = {"lml": -inf, "a": 0, "v0": 0, "v1": 0, "beta": 0}
for a in [0.1, 0.5, 0.9]:
    # cov(y) = v0*(aΣ + (1-a)K) + v1*I
    QS = economic_qs(a * Sigma + (1 - a) * K)
    lmm = LMM(y, X, QS)
    lmm.fit(verbose=False)
    if lmm.lml() > best["lml"]:
        best["lml"] = lmm.lml()
        best["a"] = a
        best["v0"] = lmm.v0
        best["v1"] = lmm.v1
        best["alpha"] = lmm.beta

# The way LMM represents: 𝓝(y|Xb, scale * ((1-δ)K  + δI))
# lmm.delta = 0.1
# lmm.scale = 3.4
# lmm.fix("scale")
# lmm.fix("delta")

# H0 optimal parameters
alpha = lmm.beta[:-1]
beta = lmm.beta[-1]
# e²Σ + g²K = s²(aΣ + (1-a)K)
# e² = s²*a
# g² = s²*(1-a)
s2 = lmm.v0  # s²
eps2 = lmm.v1  # 𝜀²

# H1 via score test
# Let K₀ = g²K + e²Σ + 𝜀²I
# with optimal values e² and 𝜀² found above.
K0 = lmm.covariance()

# Let P₀ = K⁻¹ - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹.
K0iX = solve(K0, X)
P0 = inv(K0) - K0iX @ solve(X.T @ K0iX, K0iX.T)

# P₀𝐲 = K⁻¹𝐲 - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹𝐲.
K0iy = solve(K0, y)
P0y = K0iy - solve(K0, X @ solve(X.T @ K0iX, X.T @ K0iy))

# The covariance matrix of H1 is K = K₀ + b²diag(𝐠)⋅Σ⋅diag(𝐠)
# We have ∂K/∂b² = diag(𝐠)⋅Σ⋅diag(𝐠)
# The score test statistics is given by
# Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
dK = ddot(g, ddot(Sigma, g))
Q = (P0y.T @ dK @ P0y) / 2

# Q is the score statistic for our interaction test and follows a linear combination
# of chi-squared (df=1) distributions:
# Q ∼ ∑λχ², where λᵢ are the non-zero eigenvalues of ½√P₀⋅∂K⋅√P₀.
sqrP0 = sqrtm(P0)
lambdas = eigvalsh((sqrP0 @ dK @ sqrP0) / 2)
lambdas = lambdas[lambdas > epsilon.small]
print(lambdas)
print(Q)
