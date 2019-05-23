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
    r"""
    Mixed-model with genetic effect heterogeneity.

    The extended StructLMM model (two random effects) is

        𝐲 = 𝙼𝛂 + 𝐠𝛽 + 𝐠⊙𝛃 + 𝐞 + 𝐮 + 𝛆,                                          (1)

    where

        (𝐠⊙𝛃)ᵢ = 𝐠ᵢ𝛃ᵢ
        𝛽 ∼ 𝓝(0, 𝓋₀⋅ρ),
        𝛃 ∼ 𝓝(𝟎, 𝓋₀(1-ρ)𝙴𝙴ᵀ),
        𝐞 ∼ 𝓝(𝟎, 𝓋₁𝚆𝚆ᵀ),
        𝐮 ~ 𝓝(𝟎, g²𝙺), and
        𝛆 ∼ 𝓝(𝟎, 𝓋₂𝙸).

    The matrices 𝙴 and 𝚆 are generally the same, and represent the environment
    configuration for each sample.
    The parameter ρ ∈ [𝟶, 𝟷] dictates the relevance of genotype-environment interaction
    versus the genotype effect alone.
    The term 𝐞 accounts for additive environment-only effects while 𝛆 accounts for
    noise effects.
    The term 𝐮 accounts for population structure.

    The above model is equivalent to

        𝐲 = 𝙼𝛂 + 𝐠⊙𝛃 + 𝐞 + 𝐮 + 𝛆,                                               (2)

    where

        𝛃 ∼ 𝓝(𝟎, 𝓋₀(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)),
        𝐞 ∼ 𝓝(𝟎, 𝓋₁𝚆𝚆ᵀ),
        𝐮 ~ 𝓝(𝟎, g²𝙺), and
        𝛆 ∼ 𝓝(𝟎, 𝓋₂𝙸).

    Notice that the 𝛃 in Eqs. (1) and (2) are not the same.
    Its marginalised form is given by

        𝐲 ∼ 𝓝(𝙼𝛂, 𝓋₀𝙳(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)𝙳 + 𝓋₁𝚆𝚆ᵀ + g²𝙺 + 𝓋₂𝙸),

    where 𝙳 = diag(𝐠).

    StructLMM method is used to perform two types of statistical tests.
    The association one compares the following hypotheses:

        𝓗₀: 𝓋₀ = 0
        𝓗₁: 𝓋₀ > 0

    𝓗₀ denotes no genetic association, while 𝓗₁ models any genetic association.
    In particular, 𝓗₁ includes genotype-environment interaction as part of genetic
    association.
    The interaction test is slightly more complicated as the term 𝐠𝛽 in Eq. (1) is now
    considered a fixed one.
    In pratice, however, we instead include 𝐠 in the covariates matrix 𝙼 and set ρ = 0
    in Eq. (2).
    We refer to this modified model as the interaction model.
    The compared hypotheses are:

        𝓗₀: 𝓋₀ = 0 (given the interaction model)
        𝓗₁: 𝓋₀ > 0 (given the interaction model)
    """

    def __init__(self, y, W, E, G=None, a_values=None, K=None):

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
        self.X = concatenate((self.W, g[:, newaxis]), axis=1)
        best = {"lml": -inf, "a": 0, "v0": 0, "v1": 0, "beta": 0}
        for a in self.a_values:
            # cov(y) = v0*(aΣ + (1-a)K) + v1*I
            lmm = LMM(self.y, self.X, self.QS_a[a], restricted=True)
            lmm.fit(verbose=False)
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
        s2 = self.best["v0"]  # s²
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
        return pval
