
import numpy as np
import scipy as sp
from numpy import concatenate, inf, newaxis
from numpy.linalg import eigvalsh, inv, solve
from numpy.random import RandomState
from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, economic_qs, economic_svd
from scipy.linalg import sqrtm
from chiscore import davies_pvalue, mod_liu, optimal_davies_pvalue

from glimix_core.lmm import LMM


class StructLMM2:
    r'''
    Mixed-model with genetic effect heterogeneity.

    The extended StructLMM model (two random effects) is ::

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
    '''
    '''
    test
    random = RandomState(0)
    n = 1000
    c = 2
    y = random.randn(n)
    W = random.randn(n, c)
    # g = random.randn(n)
    E = random.randn(n, 4)
    Sigma = E @ E.T
    X = random.randn(n, 5)
    K = X @ X.T
    '''

    def __init__(self, y, W, E, G = None, a_values = None, K = None):

        self.y = y
        self.E = E
        self.G = G
        self.W = W

        self.Sigma = E @ E.T

        if self.G is None:
            self.K = np.eye(self.y.shape[0])
        else:
            self.K = G @ G.T

        self.a_values = a_values
        if self.a_values is None:
            self.a_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.Cov = {}
        self.QS_a = {}
        for a in self.a_values:
            self.Cov[a] = a * self.Sigma + (1 - a) * self.K
            self.QS_a[a] = economic_qs(self.Cov[a])


    def fit_null(self, g):
        self.X = concatenate((self.W, g[:, newaxis]), axis = 1)
        best = {"lml": -inf, "a": 0, "v0": 0, "v1": 0, "beta": 0}
        for a in self.a_values:
        # cov(y) = v0*(aΣ + (1-a)K) + v1*I
            lmm = LMM(self.y, self.X, self.QS_a[a], restricted = True)
            lmm.fit(verbose = False)
            if lmm.lml() > best["lml"]:
                best["lml"] = lmm.lml()
                best["a"] = a
                best["v0"] = lmm.v0
                best["v1"] = lmm.v1
                best["alpha"] = lmm.beta
                best["covariance"] = lmm.covariance()
        self.best = best


    def score_2_dof(self, g):
        alpha = self.best["alpha"][:-1]
        beta = self.best["alpha"][-1]
        # e²Σ + g²K = s²(aΣ + (1-a)K)
        # e² = s²*a
        # g² = s²*(1-a)
        s2 = self.best["v0"] # s²
        eps2 = self.best["v1"]  # 𝜀²

        # H1 via score test
        # Let K₀ = g²K + e²Σ + 𝜀²I
        # with optimal values e² and 𝜀² found above.
        K0 = self.best["covariance"]

        # Let P₀ = K⁻¹ - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹.
        K0iX = solve(K0, self.X)
        P0 = inv(K0) - K0iX @ solve(self.X.T @ K0iX, K0iX.T)

        # P₀𝐲 = K⁻¹𝐲 - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹𝐲.
        K0iy = solve(K0, self.y)
        P0y = K0iy - solve(K0, self.X @ solve(self.X.T @ K0iX, self.X.T @ K0iy))

        # The covariance matrix of H1 is K = K₀ + b²diag(𝐠)⋅Σ⋅diag(𝐠)
        # We have ∂K/∂b² = diag(𝐠)⋅Σ⋅diag(𝐠)
        # The score test statistics is given by
        # Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
        dK = ddot(g, ddot(self.Sigma, g))
        Q = (P0y.T @ dK @ P0y) / 2

        # Q is the score statistic for our interaction test and follows a linear combination
        # of chi-squared (df=1) distributions:
        # Q ∼ ∑λχ², where λᵢ are the non-zero eigenvalues of ½√P₀⋅∂K⋅√P₀.
        sqrP0 = sqrtm(P0)
        # lambdas = eigvalsh((sqrP0 @ dK @ sqrP0) / 2)
        # lambdas = lambdas[lambdas > epsilon.small]
        # print(lambdas)
        # print(Q)
        pval = davies_pvalue(Q, (sqrP0 @ dK @ sqrP0) / 2)
        return(pval)





