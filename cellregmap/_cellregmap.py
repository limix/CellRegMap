from typing import Optional

from glimix_core.lmm import LMM
from numpy import (
    asarray,
    atleast_1d,
    atleast_2d,
    concatenate,
    inf,
    linspace,
    sqrt,
    stack,
)
from numpy_sugar import ddot
from numpy_sugar.linalg import economic_qs_linear
from tqdm import tqdm

from ._math import PMat, QSCov, ScoreStatistic


class CellRegMap:
    """
    Mixed-model with genetic effect heterogeneity.

    The CellRegMap model can be cast as:

       𝐲 = W𝛂 + 𝐠𝛽₁ + 𝐠⊙𝛃₂ + 𝐞 + 𝐮 + 𝛆,                                             (1)

    where:

        𝛃₂ ~ 𝓝(𝟎, 𝓋₃𝙴₀𝙴₀ᵀ),
        𝐞 ~ 𝓝(𝟎, 𝓋₁ρ₁𝙴₁𝙴₁ᵀ),
        𝐮 ~ 𝓝(𝟎, 𝓋₁(1-ρ₁)𝙺⊙𝙴₂𝙴₂ᵀ), and
        𝛆 ~ 𝓝(𝟎, 𝓋₂𝙸).

    𝐠⊙𝛃 is a randome effect term which models the GxE effect.
    Additionally, W𝛂 models additive covariates and 𝐠𝛽₁ models persistent genetic effects.
    Both are modelled as fixed effects.
    On the other hand, 𝐞, 𝐮 and 𝛆 are modelled as random effects
    𝐞 is the environment effect, 𝐮 is a background term accounting for interactions between population structure
    and environmental structure, and 𝛆 is the iid noise.
    The full covariance of 𝐲 is therefore given by:

        cov(𝐲) = 𝓋₃𝙳𝙴₀𝙴₀ᵀ𝙳 + 𝓋₁ρ₁𝙴₁𝙴₁ᵀ + 𝓋₁(1-ρ₁)𝙺⊙𝙴₂𝙴₂ᵀ + 𝓋₂𝙸,

    where 𝙳 = diag(𝐠). Its marginalised form is given by:

        𝐲 ~ 𝓝(W𝛂 + 𝐠𝛽₁, 𝓋₃𝙳𝙴₀𝙴₀ᵀ𝙳 + 𝓋₁(ρ₁𝙴₁𝙴₁ᵀ + (1-ρ₁)𝙺⊙𝙴₂𝙴₂ᵀ) + 𝓋₂𝙸).

    sc-StructLMM method is used to perform an interaction test:

    The interaction test compares the following hypotheses (from Eq. 1):

        𝓗₀: 𝓋₃ = 0
        𝓗₁: 𝓋₃ > 0

    𝓗₀ denotes no GxE effects, while 𝓗₁ models the presence of GxE effects.

    """

    def __init__(self, y, W, E, Ls=[], E0=None, E1=None):
        self._y = asarray(y, float).flatten()
        self._W = asarray(W, float)

        if E is None:
            assert E0 is not None
            assert E1 is not None
            self._E0 = asarray(E0, float)
            self._E1 = asarray(E1, float)
        else:
            self._E0 = asarray(E, float)
            self._E1 = asarray(E, float)
        self._Ls = list(asarray(L, float) for L in Ls)

        assert self._W.ndim == 2
        assert self._E0.ndim == 2
        assert self._E1.ndim == 2

        assert self._y.shape[0] == self._W.shape[0]
        assert self._y.shape[0] == self._E0.shape[0]
        assert self._y.shape[0] == self._E1.shape[0]
        for L in Ls:
            assert self._y.shape[0] == L.shape[0]
            assert L.ndim == 2

        self._null_lmm_assoc = {}

        self._halfSigma = {}
        self._Sigma_qs = {}
        # TODO: remove it after debugging
        self._Sigma = {}

        if len(Ls) == 0:
            # self._rho0 = [1.0]
            self._rho1 = [1.0]
            self._halfSigma[1.0] = self._E1
            self._Sigma_qs[1.0] = economic_qs_linear(self._E1, return_q1=False)
        else:
            # self._rho0 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            self._rho1 = linspace(0, 1, 11)
            for rho1 in self._rho1:
                # Σ = ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺⊙E
                # concatenate((sqrt(rho1) * self._E, sqrt(1 - rho1) * G1), axis=1)
                # self._Sigma[rho1] = rho1 * self._EE + (1 - rho1) * self._K
                # self._Sigma_qs[rho1] = economic_qs(self._Sigma[rho1])
                a = sqrt(rho1)
                b = sqrt(1 - rho1)
                hS = concatenate([a * self._E1] + [b * L for L in Ls], axis=1)
                self._halfSigma[rho1] = hS
                self._Sigma_qs[rho1] = economic_qs_linear(
                    self._halfSigma[rho1], return_q1=False
                )

    @property
    def n_samples(self):
        return self._y.shape[0]

    def predict_interaction(self, G, MAF):
        """
        Share screen.
        """
        # breakpoint()
        G = asarray(G, float)
        E0 = self._E0
        W = self._W
        n_snps = G.shape[1]
        beta_g_s = []
        beta_gxe_s = []

        p = asarray(atleast_1d(MAF), float)
        normalization = 1 / sqrt(2 * p * (1 - p))

        for i in range(n_snps):
            g = G[:, [i]]
            # mean(𝐲) = W𝛂 + 𝐠𝛽₁ + 𝙴𝝲 = 𝙼𝛃
            M = concatenate((W, g, E0), axis=1)
            gE = g * E0
            best = {"lml": -inf, "rho1": 0}
            hSigma_p = {}
            Sigma_qs = {}
            for rho1 in self._rho1:
                # Σ[ρ₁] = ρ₁(𝐠⊙𝙴)(𝐠⊙𝙴)ᵀ + (1-ρ₁)𝙺⊙EEᵀ
                a = sqrt(rho1)
                b = sqrt(1 - rho1)
                hSigma_p[rho1] = concatenate(
                    [a * gE] + [b * L for L in self._Ls], axis=1
                )
                # (
                #     (a * gE, b * self._G), axis=1
                # )
                # cov(𝐲) = 𝓋₁Σ[ρ₁] + 𝓋₂𝙸
                # lmm = Kron2Sum(Y, [[1]], M, hSigma_p[rho1], restricted=True)
                Sigma_qs[rho1] = economic_qs_linear(
                    hSigma_p[rho1], return_q1=False
                )
                lmm = LMM(self._y, M, Sigma_qs[rho1], restricted=True)
                lmm.fit(verbose=False)

                if lmm.lml() > best["lml"]:
                    best["lml"] = lmm.lml()
                    best["rho1"] = rho1
                    best["lmm"] = lmm

            # breakpoint()
            lmm = best["lmm"]
            # beta_g = 𝛽₁
            beta_g = lmm.beta[W.shape[1]]
            # yadj = 𝐲 - 𝙼𝛃
            yadj = (self._y - lmm.mean()).reshape(self._y.shape[0], 1)
            rho1 = best["rho1"]
            v1 = lmm.v0
            v2 = lmm.v1
            hSigma_p_qs = economic_qs_linear(hSigma_p[rho1], return_q1=False)
            qscov = QSCov(hSigma_p_qs[0][0], hSigma_p_qs[1], v1, v2)
            # v = cov(𝐲)⁻¹(𝐲 - 𝙼𝛃)
            v = qscov.solve(yadj)

            sigma2_gxe = v1 * rho1
            beta_gxe = sigma2_gxe * E0 @ (gE.T @ v) * normalization[i]
            # beta_star = (beta_g * normalization + beta_gxe)

            beta_g_s.append(beta_g)
            beta_gxe_s.append(beta_gxe)


        return (asarray(beta_g_s), stack(beta_gxe_s).T)

    def estimate_aggregate_environment(self, g):
        g = atleast_2d(g).reshape((g.size, 1))
        E0 = self._E0
        gE = g * E0
        W = self._W
        M = concatenate((W, g, E0), axis=1)
        best = {"lml": -inf, "rho1": 0}
        hSigma_p = {}
        for rho1 in self._rho1:
            # Σₚ = ρ₁(𝐠⊙𝙴)(𝐠⊙𝙴)ᵀ + (1-ρ₁)𝙺⊙E
            a = sqrt(rho1)
            b = sqrt(1 - rho1)
            hSigma_p[rho1] = concatenate([a * gE] + [b * L for L in self._Ls], axis=1)
            # cov(𝐲) = 𝓋₁Σₚ + 𝓋₂𝙸
            # lmm = Kron2Sum(Y, [[1]], M, hSigma_p[rho1], restricted=True)
            QS = self._Sigma_qs[rho1]
            lmm = LMM(self._y, M, QS, restricted=True)
            lmm.fit(verbose=False)

            if lmm.lml() > best["lml"]:
                best["lml"] = lmm.lml()
                best["rho1"] = rho1
                best["lmm"] = lmm

        lmm = best["lmm"]
        yadj = self._y - lmm.mean()
        # rho1 = best["rho1"]
        v1 = lmm.v0
        v2 = lmm.v1
        rho1 = best["rho1"]
        sigma2_gxe = rho1 * v1
        hSigma_p_qs = economic_qs_linear(hSigma_p[rho1], return_q1=False)
        qscov = QSCov(hSigma_p_qs[0][0], hSigma_p_qs[1], v1, v2)
        # v = cov(𝐲)⁻¹yadj
        v = qscov.solve(yadj)
        beta_gxe = sigma2_gxe * gE.T @ v

        return E0 @ beta_gxe

    def scan_interaction(
        self, G, idx_E: Optional[any] = None, idx_G: Optional[any] = None
    ):
        """
        𝐲 = W𝛂 + 𝐠𝛽₁ + 𝐠⊙𝛃₂ + 𝐞 + 𝐮 + 𝛆
           [fixed=X]   [H1]

        𝛃₂ ~ 𝓝(𝟎, 𝓋₃𝙴₀𝙴₀ᵀ),
        𝐞 ~ 𝓝(𝟎, 𝓋₁ρ₁𝙴₁𝙴₁ᵀ),
        𝐮 ~ 𝓝(𝟎, 𝓋₁(1-ρ₁)𝙺⊙𝙴₂𝙴₂ᵀ), and
        𝛆 ~ 𝓝(𝟎, 𝓋₂𝙸).

        𝓗₀: 𝓋₃ = 0
        𝓗₁: 𝓋₃ > 0
        """
        # TODO: make sure G is nxp
        from chiscore import davies_pvalue

        G = asarray(G, float)
        n_snps = G.shape[1]
        pvalues = []
        info = {"rho1": [], "e2": [], "g2": [], "eps2": []}

        for i in tqdm(range(n_snps)):
            g = G[:, [i]]
            X = concatenate((self._W, g), axis=1)
            best = {"lml": -inf, "rho1": 0}
            # Null model fitting: find best (𝛂, 𝛽₁, 𝓋₁, 𝓋₂, ρ₁)
            for rho1 in self._rho1:
                # QS = self._Sigma_qs[rho1]
                # halfSigma = self._halfSigma[rho1]
                # Σ = ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺⊙E
                # cov(y₀) = 𝓋₁Σ + 𝓋₂I
                QS = self._Sigma_qs[rho1]
                lmm = LMM(self._y, X, QS, restricted=True)
                lmm.fit(verbose=False)

                if lmm.lml() > best["lml"]:
                    best["lml"] = lmm.lml()
                    best["rho1"] = rho1
                    best["lmm"] = lmm

            lmm = best["lmm"]
            # H1 via score test
            # Let K₀ = e²𝙴𝙴ᵀ + g²𝙺⊙E + 𝜀²I
            # e²=𝓋₁ρ₁
            # g²=𝓋₁(1-ρ₁)
            # 𝜀²=𝓋₂
            # with optimal values 𝓋₁ and 𝓋₂ found above.
            info["rho1"].append(best["rho1"])
            info["e2"].append(lmm.v0 * best["rho1"])
            info["g2"].append(lmm.v0 * (1 - best["rho1"]))
            info["eps2"].append(lmm.v1)
            # QS = economic_decomp( Σ(ρ₁) )
            Q0 = self._Sigma_qs[best["rho1"]][0][0]
            S0 = self._Sigma_qs[best["rho1"]][1]
            # e2 = best["lmm"].v0 * best["rho1"]
            # g2 = best["lmm"].v0 * (1 - best["rho1"])
            # eps2 = best["lmm"].v1
            # EE = self._E @ self._E.T
            # K = self._G @ self._G.T
            # K0 = e2 * EE + g2 * K + eps2 * eye(K.shape[0])
            qscov = QSCov(
                Q0,
                S0,
                lmm.v0,  # 𝓋₁
                lmm.v1,  # 𝓋₂
            )
            # start = time()
            # qscov = QSCov(self._Sigma_qs[best["rho1"]], lmm.C0[0, 0], lmm.C1[0, 0])
            # print(f"Elapsed: {time() - start}")
            # X = concatenate((self._E, g), axis=1)
            X = concatenate((self._W, g), axis=1)

            # Let P₀ = K₀⁻¹ - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹.
            P = PMat(qscov, X)
            # P0 = inv(K0) - inv(K0) @ X @ inv(X.T @ inv(K0) @ X) @ X.T @ inv(K0)

            # P₀𝐲 = K₀⁻¹𝐲 - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹𝐲.

            # Useful for permutation
            if idx_E is None:
                E0 = self._E0
            else:
                E0 = self._E0[idx_E, :]

            # The covariance matrix of H1 is K = K₀ + 𝓋₃diag(𝐠)⋅𝙴𝙴ᵀ⋅diag(𝐠)
            # We have ∂K/∂𝓋₃ = diag(𝐠)⋅𝙴𝙴ᵀ⋅diag(𝐠)
            # The score test statistics is given by
            # Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
            # start = time()

            # Useful for permutation
            if idx_G is None:
                gtest = g.ravel()
            else:
                gtest = g.ravel()[idx_G]

            ss = ScoreStatistic(P, qscov, ddot(gtest, E0))
            Q = ss.statistic(self._y)
            # import numpy as np

            # deltaK = np.diag(gtest) @ EE @ np.diag(gtest)
            # Q_ = 0.5 * self._y.T @ P0 @ deltaK @ P0 @ self._y
            # print(f"Elapsed: {time() - start}")
            # Q is the score statistic for our interaction test and follows a linear
            # combination
            # of chi-squared (df=1) distributions:
            # Q ∼ ∑λχ², where λᵢ are the non-zero eigenvalues of ½√P₀⋅∂K⋅√P₀.
            # Since eigenvals(𝙰𝙰ᵀ) = eigenvals(𝙰ᵀ𝙰) (TODO: find citation),
            # we can compute ½(√∂K)P₀(√∂K) instead.
            # start = time()
            # import scipy as sp
            # sqrtm = sp.linalg.sqrtm
            # np.linalg.eigvalsh(0.5 * sqrtm(P0) @ deltaK @ sqrtm(P0))
            # np.linalg.eigvalsh(0.5 * sqrtm(deltaK) @ P0 @ sqrtm(deltaK))
            # TODO: compare with Liu approximation, maybe try a computational intensive
            # method
            pval, pinfo = davies_pvalue(Q, ss.matrix_for_dist_weights(), True)
            pvalues.append(pval)
            # print(f"Elapsed: {time() - start}")

        info = {key: asarray(v, float) for key, v in info.items()}
        return asarray(pvalues, float), info
