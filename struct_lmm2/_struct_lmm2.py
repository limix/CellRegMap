from glimix_core.lmm import LMM
from numpy import asarray, concatenate, diag, empty, inf, ones, sqrt, stack, trace
from numpy.linalg import eigvalsh, inv, solve
from numpy_sugar import ddot
from numpy_sugar.linalg import economic_qs
from scipy.linalg import sqrtm

from ._math import (
    P_matrix,
    qmin,
    score_statistic,
    score_statistic_distr_weights,
    score_statistic_liu_params,
)


class StructLMM2:
    r"""
    Mixed-model with genetic effect heterogeneity.

    The extended StructLMM model (two random effects) is:

        𝐲 = W𝛂 + 𝐠⊙𝛃 + 𝐞 + 𝐮 + 𝛆,                                               (1)

    where:

        𝛃 ∼ 𝓝(𝟎, 𝓋₀((1-ρ₀)𝟏𝟏ᵀ + ρ₀𝙴𝙴ᵀ)),
        𝐞 ∼ 𝓝(𝟎, 𝓋₁ρ₁EEᵀ),
        𝐮 ∼ 𝓝(𝟎, 𝓋₁(1-ρ₁)𝙺), and
        𝛆 ∼ 𝓝(𝟎, 𝓋₂𝙸).

    𝐠⊙𝛃 is made of two components: the persistent genotype effect and the GxE effect.
    𝐞 is the environment effect, 𝐮 is the population structure effect, and 𝛆 is the iid
    noise. The full covariance of 𝐲 is therefore given by:

        cov(𝐲) = 𝓋₀(1-ρ₀)𝙳𝟏𝟏ᵀ𝙳 + 𝓋₀ρ₀𝙳𝙴𝙴ᵀ𝙳 + 𝓋₁ρ₁EEᵀ + 𝓋₁(1-ρ₁)𝙺 + 𝓋₂𝙸.

    Its marginalised form is given by:

        𝐲 ∼ 𝓝(W𝛂, 𝓋₀𝙳((1-ρ₀)𝟏𝟏ᵀ + ρ₀𝙴𝙴ᵀ)𝙳 + 𝓋₁(ρ₁EEᵀ + (1-ρ₁)𝙺) + 𝓋₂𝙸),

    where 𝙳 = diag(𝐠).

    StructLMM method is used to perform two types of statistical tests.

    1. The association test compares the following hypotheses (from Eq.1):

        𝓗₀: 𝓋₀ = 0
        𝓗₁: 𝓋₀ > 0

    𝓗₀ denotes no genetic association, while 𝓗₁ models any genetic association.
    In particular, 𝓗₁ includes genotype-environment interaction as part of genetic
    association.

    2. The interaction test is slighlty different as the persistent genotype
    effect is now considered to be a fixed effect, and added to the model as an
    additional covariate term:

        𝐲 = W𝛂 + 𝐠𝛽₁ + 𝐠⊙𝛃₂ + 𝐞 + 𝐮 + 𝛆,                                        (2)

    where:

        𝛃₂ ∼ 𝓝(𝟎, 𝓋₃𝙴𝙴ᵀ),
        𝐞 ∼ 𝓝(𝟎, 𝓋₁ρ₁𝙴𝙴ᵀ),
        𝐮 ~ 𝓝(𝟎, 𝓋₁(1-ρ₁)𝙺), and
        𝛆 ∼ 𝓝(𝟎, 𝓋₂𝙸).

    We refer to this modified model as the interaction model.
    The compared hypotheses are:

        𝓗₀: 𝓋₃ = 0
        𝓗₁: 𝓋₃ > 0
    """

    def __init__(self, y, W, E, K=None):
        self._y = y
        self._W = W
        self._E = E
        self._K = K
        self._EE = E @ E.T

        self._rho0 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self._rho1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self._null_lmm_assoc = {}

        self._Sigma = {}
        self._Sigma_qs = {}
        for rho1 in self._rho1:
            # Σ = ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺
            self._Sigma[rho1] = rho1 * self._EE + (1 - rho1) * self._K
            self._Sigma_qs[rho1] = economic_qs(self._Sigma[rho1])

    def fit_null_association(self):
        best = {"lml": -inf, "lmm": None, "rho1": -1.0}
        for rho1, Sigma_qs in self._Sigma_qs.items():
            lmm = LMM(self._y, self._W, Sigma_qs, restricted=True)
            lmm.fit(verbose=True)
            lml = lmm.lml()
            if lml > best["lml"]:
                best["lml"] = lml
                best["lmm"] = lmm
                best["rho1"] = rho1
        self._null_lmm_assoc = best

    def scan_association(self, G):
        n_snps = G.shape[1]
        lmm = self._null_lmm_assoc["lmm"]
        K0 = lmm.covariance()
        P = P_matrix(self._W, K0)
        # H1 vs H0 via score test
        for i in range(n_snps):
            g = G[:, i].reshape(G.shape[0], 1)
            K0 = lmm.covariance()
            weights = []
            liu_params = []
            for rho0 in self._rho0:
                # K = K0 + s2total * (
                #     (1 - rho0) * g @ g.T + rho0 * diag(g) @ self._EE @ diag(g)
                # )
                dK = (1 - rho0) * g @ g.T + rho0 * diag(g) @ self._EE @ diag(g)
                # 𝙿 = 𝙺⁻¹ - 𝙺⁻¹𝚆(𝚆ᵀ𝙺⁻¹𝚆)⁻¹𝚆ᵀ𝙺⁻¹
                # 𝑄 = ½𝐲ᵀ𝙿(∂𝙺)𝙿𝐲.
                Q = score_statistic(self._y, self._W, K0, dK)
                weights += [score_statistic_distr_weights(self._W, K0, dK)]
                liu_params += [score_statistic_liu_params(Q, weights)]

            q = qmin(liu_params)

            # 3. Calculate quantities that occur in null distribution
            # g has to be a column-vector
            D = diag(g.ravel())
            Pg = P @ g
            m = (g.T @ Pg)[0, 0]
            M = 1 / m * (sqrtm(P) @ g @ g.T @ sqrtm(P))
            H1 = E.T @ D.T @ P @ D @ E
            H2 = E.T @ D.T @ sqrtm(P) @ M @ sqrtm(P) @ D @ E
            H = H1 - H2
            lambdas = eigvalsh(H / 2)
            lambdas = eigh

            eta = ETxPx11xPxE @ ZTIminusMZ
            vareta = 4 * trace(eta)

            OneZTZE = 0.5 * (g.T @ PxoE)
            tau_top = OneZTZE @ OneZTZE.T
            tau_rho = empty(len(self._rho0))
            for i in range(len(self._rho0)):
                tau_rho[i] = self._rho0[i] * m + (1 - self._rho0[i]) / m * tau_top

            MuQ = sum(eigh)
            VarQ = sum(eigh ** 2) * 2 + vareta
            KerQ = sum(eigh ** 4) / (sum(eigh ** 2) ** 2) * 12
            Df = 12 / KerQ

    def scan_interaction(self, G):
        from chiscore import davies_pvalue

        n_snps = G.shape[1]
        pvalues = []
        for i in range(n_snps):
            g = G[:, [i]]
            Wg = concatenate((self._W, g), axis=1)
            best = {"lml": -inf, "a": 0, "v0": 0, "v1": 0, "beta": 0}
            for a in self._rho1:
                QS = self._Sigma_qs[a]
                # cov(y) = v0*(aΣ + (1-a)K) + v1*Is
                lmm = LMM(self._y, Wg, QS, restricted=True)
                lmm.fit(verbose=False)
                if lmm.lml() > best["lml"]:
                    best["lml"] = lmm.lml()
                    best["a"] = a
                    best["v0"] = lmm.v0
                    best["v1"] = lmm.v1
                    best["alpha"] = lmm.beta
                    best["lmm"] = lmm

            lmm = best["lmm"]
            "H1 via score test"
            # Let K₀ = g²K + e²Σ + 𝜀²I
            # with optimal values e² and 𝜀² found above.
            K0 = lmm.covariance()
            X = concatenate((self._E, g), axis=1)

            # Let P₀ = K⁻¹ - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹.
            K0iX = solve(K0, X)
            P0 = inv(K0) - K0iX @ solve(X.T @ K0iX, K0iX.T)

            # P₀𝐲 = K⁻¹𝐲 - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹𝐲.
            K0iy = solve(K0, self._y)
            P0y = K0iy - solve(K0, X @ solve(X.T @ K0iX, X.T @ K0iy))

            # The covariance matrix of H1 is K = K₀ + b²diag(𝐠)⋅Σ⋅diag(𝐠)
            # We have ∂K/∂b² = diag(𝐠)⋅Σ⋅diag(𝐠)
            # The score test statistics is given by
            # Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
            dK = ddot(g.ravel(), ddot(self._EE, g.ravel()))
            Q = (P0y.T @ dK @ P0y) / 2

            # Q is the score statistic for our interaction test and follows a linear combination
            # of chi-squared (df=1) distributions:
            # Q ∼ ∑λχ², where λᵢ are the non-zero eigenvalues of ½√P₀⋅∂K⋅√P₀.
            sqrP0 = sqrtm(P0)
            pval = davies_pvalue(Q, (sqrP0 @ dK @ sqrP0) / 2)
            pvalues.append(pval)

        return asarray(pvalues, float)
