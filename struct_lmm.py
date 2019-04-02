"""
Mixed-model with genetic effect heterogeneity.

The StructLMM model is ::

    𝐲 = W𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + 𝛆,

where ::

    𝐠⊙𝛃 = ∑ᵢ𝐠ᵢ𝛽ᵢ
    𝛃 ∼ 𝓝(𝟎, b²Σ)
    𝐞 ∼ 𝓝(𝟎, e²Σ)
    𝛆 ∼ 𝓝(𝟎, 𝜀²I)
    Σ = EEᵀ

If one considers 𝛽 ∼ 𝓝(0, p²), we can insert
𝛽 into 𝛃 ::

    𝛃_ ∼ 𝓝(𝟎, p²𝟏𝟏ᵀ + b²Σ)

"""
from numpy import concatenate, newaxis
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

H0: b² = 0 => 𝐲 = W𝛂 + 𝐠𝛽 + 𝐞 + 𝛆
    𝐲 ∼ 𝓝(W𝛂 + 𝐠𝛽, e²Σ + 𝜀²I)
H1: b² > 0 => 𝐲 = W𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + 𝛆
    𝐲 ∼ 𝓝(W𝛂 + 𝐠𝛽, e²Σ + 𝜀²I + b²Σ)
"""
X = concatenate((W, g[:, newaxis]), axis=1)
lmm = LMM(y, X, QS)
lmm.fit(verbose=False)

# H0 optimal parameters
alpha = lmm.beta[:-1]
beta = lmm.beta[-1]
e2 = lmm.v0  # e²
eps2 = lmm.v1  # 𝜀²

# H1 via score test
# Let K₀ = e²Σ + 𝜀²I with optimal values e² and 𝜀² found above.
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
