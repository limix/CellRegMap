from typing import Optional

from tqdm import tqdm
from chiscore import optimal_davies_pvalue
from glimix_core.lmm import LMM, Kron2Sum
from numpy import (
    asarray,
    concatenate,
    diag,
    empty,
    eye,
    inf,
    newaxis,
    ones,
    sqrt,
    stack,
    trace,
    vstack,
    linspace,
)

from numpy.linalg import eigvalsh, inv, lstsq, multi_dot
from scipy.linalg import sqrtm

from numpy_sugar import ddot
from numpy_sugar.linalg import trace2, economic_qs_linear

from ._math import (
    P_matrix,
    PMat,
    QSCov,
    ScoreStatistic,
    # economic_qs_linear,
    qmin,
    rsolve,
    score_statistic,
    score_statistic_distr_weights,
    score_statistic_liu_params,
    score_statistic_qs,
)


class StructLMM2:
    """
    Mixed-model with genetic effect heterogeneity.

    The extended StructLMM model (two random effects) is:

        𝐲 = W𝛂 + 𝐠⊙𝛃 + 𝐞 + 𝐮 + 𝛆,                                              (1)

    where:

        𝛃 ~ 𝓝(𝟎, 𝓋₀((1-ρ₀)𝟏𝟏ᵀ + ρ₀𝙴𝙴ᵀ)),
        𝐞 ~ 𝓝(𝟎, 𝓋₁ρ₁𝙴𝙴ᵀ),
        𝐮 ~ 𝓝(𝟎, 𝓋₁(1-ρ₁)𝙺), and
        𝛆 ~ 𝓝(𝟎, 𝓋₂𝙸).

    𝐠⊙𝛃 is made of two components: the persistent genotype effect and the GxE effect. 𝐞 is the
    environment effect, 𝐮 is the population structure effect, and 𝛆 is the iid noise. The full
    covariance of 𝐲 is therefore given by:

        cov(𝐲) = 𝓋₀(1-ρ₀)𝙳𝟏𝟏ᵀ𝙳 + 𝓋₀ρ₀𝙳𝙴𝙴ᵀ𝙳 + 𝓋₁ρ₁𝙴𝙴ᵀ + 𝓋₁(1-ρ₁)𝙺 + 𝓋₂𝙸,

    where 𝙳 = diag(𝐠). Its marginalised form is given by:

        𝐲 ~ 𝓝(W𝛂, 𝓋₀𝙳((1-ρ₀)𝟏𝟏ᵀ + ρ₀𝙴𝙴ᵀ)𝙳 + 𝓋₁(ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺) + 𝓋₂𝙸).

    StructLMM method is used to perform two types of statistical tests.

    1. The association test compares the following hypotheses (from Eq. 1):

        𝓗₀: 𝓋₀ = 0
        𝓗₁: 𝓋₀ > 0

    𝓗₀ denotes no genetic association, while 𝓗₁ models any genetic association. In particular, 𝓗₁
    includes genotype-environment interaction as part of genetic association.

    2. The interaction test is slighlty different as the persistent genotype effect is now
    considered to be a fixed effect, and added to the model as an additional covariate term:

        𝐲 = W𝛂 + 𝐠𝛽₁ + 𝐠⊙𝛃₂ + 𝐞 + 𝐮 + 𝛆,                                       (2)

    where:

        𝛃₂ ~ 𝓝(𝟎, 𝓋₃𝙴𝙴ᵀ),
        𝐞  ~ 𝓝(𝟎, 𝓋₁ρ₁𝙴𝙴ᵀ),
        𝐮  ~ 𝓝(𝟎, 𝓋₁(1-ρ₁)𝙺), and
        𝛆  ~ 𝓝(𝟎, 𝓋₂𝙸).

    We refer to this modified model as the interaction model. The compared hypotheses in this case
    are:

        𝓗₀: 𝓋₃ = 0
        𝓗₁: 𝓋₃ > 0
    """

    def __init__(self, y, W, E, G=None):
        # TODO: convert y to nx0
        # TODO: convert W to nxp
        # TODO: convert to array of floats
        self._y = y
        self._W = W
        self._E = E
        self._G = G
        # self._EE = E @ E.T

        self._null_lmm_assoc = {}

        self._halfSigma = {}
        self._Sigma_qs = {}
        # TODO: remove it after debugging
        self._Sigma = {}

        if G is None:
            self._rho0 = [1.0]
            self._rho1 = [1.0]
            self._halfSigma[1.0] = self._E
            self._Sigma_qs[1.0] = economic_qs_linear(self._E, return_q1=False)
        else:
            self._rho0 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # self._rho1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            self._rho1 = linspace(0, 1)
            for rho1 in self._rho1:
                # Σ = ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺
                # concatenate((sqrt(rho1) * self._E, sqrt(1 - rho1) * G1), axis=1)
                # self._Sigma[rho1] = rho1 * self._EE + (1 - rho1) * self._K
                # self._Sigma_qs[rho1] = economic_qs(self._Sigma[rho1])
                hS = concatenate((sqrt(rho1) * self._E, sqrt(1 - rho1) * G), axis=1)
                self._halfSigma[rho1] = hS
                self._Sigma_qs[rho1] = economic_qs_linear(
                    self._halfSigma[rho1], return_q1=False
                )
                self._Sigma[rho1] = rho1 * self._E @ self._E.T + (1 - rho1) * G @ G.T

    @property
    def _n_samples(self):
        return self._y.shape[0]

    def fit_null_association(self):
        """
        Fit p(𝐲) of Eq. (1) under the null hypothesis, 𝓋₀ = 0.

        Estimates the parameters 𝛂, 𝓋₁, ρ₁, and 𝓋₂ of:

            𝐲 ~ 𝓝(W𝛂, 𝓋₁(ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺) + 𝓋₂𝙸),

        using the restricted maximum likelihood approach.
        """
        best = {"lml": -inf, "lmm": None, "rho1": -1.0}
        for rho1, halfSigma in self._halfSigma.items():
            # for rho1, Sigma_qs in self._Sigma_qs.items():
            # Sigma_qs = self._Sigma_qs[rho1]
            # lmm2 = LMM(self._y, self._W, Sigma_qs, restricted=True)
            # lmm2.fit(verbose=False)
            lmm = Kron2Sum(
                self._y[:, newaxis], [[1]], self._W, halfSigma, restricted=True
            )
            lmm.fit(verbose=True)
            lml = lmm.lml()
            if lml > best["lml"]:
                best["lml"] = lml
                best["lmm"] = lmm
                best["rho1"] = rho1

        rho1 = best["rho1"]
        qscov = QSCov(
            self._Sigma_qs[rho1][0][0],
            self._Sigma_qs[rho1][1],
            best["lmm"].C0[0, 0],
            best["lmm"].C1[0, 0],
        )

        self._null_lmm_assoc = {
            "lml": best["lml"],
            "alpha": best["lmm"].beta,
            "v1": best["lmm"].C0[0, 0],
            "rho1": best["rho1"],
            "v2": best["lmm"].C1[0, 0],
            "qscov": qscov,
        }

    def scan_association(self, G):
        # WARNING: this method is not working yet
        """
        Association test.

        Let us define:

            𝙺₀ = 𝓋₁(ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺) + 𝓋₂𝙸.

        The marginalised form of Eq. (1) can be written as

            𝐲 ~ 𝓝(W𝛂, 𝙺₁ = 𝓋₀𝙳((1-ρ₀)𝟏𝟏ᵀ + ρ₀𝙴𝙴ᵀ)𝙳 + 𝙺₀),

        where 𝙳 = diag(𝐠). For a given ρ₀, the score test allows us to compare the hypotheses:

            𝓗₀: 𝓋₀ = 0
            𝓗₁: 𝓋₀ > 0

        by first estimating the parameters 𝛂, 𝓋₁, ρ₁, and 𝓋₂ with 𝓋₀ set to zero and then defining
        the score statistic 𝑄ᵨ = ½𝐲ᵀ𝙿(∂𝙺₁)𝙿𝐲. Under the null hypothesis, the score statistic follows
        the distribution:

            𝑄ᵨ ∼ ∑ᵢ𝜆ᵢχ²(1),

        where 𝜆ᵢ are the non-zero eigenvalues of ½√𝙿(∂𝙺₁)√𝙿 (given ρ=ρ₀).

        Unfortunately we don't know the value of ρ₀, and therefore the vanilla score test cannot be
        applied. We instead employ an alternative test defined as follows.

        - Calculate qᵨ = ½𝐲ᵀ𝙿(∂𝙺₁)𝙿𝐲 for a set of ρ₀ values. Let pᵨ be its corresponding p-value.
        - Define the T statistic as T = min{pᵨ}.
        - Derive the distribution of T under the null hypothesis that 𝓋₀=0.
        - Compute the p-value of T.

        The p-value of T will be therefore used to assess whether we have enough evidence to reject
        the hypothesis that 𝐠 has no effect.

        T statistic
        -----------

        It can be show that:

            Qᵨ ∼ ½𝜏ᵨ⋅η₀ + ½ρ𝑘,

        where:

            𝜏ᵨ = 𝑚(1-ρ₀) + (ρ₀/𝑚)𝟏ᵀ𝚉𝚉ᵀ𝙴𝙴ᵀ𝚉ᵀ𝚉𝟏
            η₀ = χ²(𝟷)
            𝙼  = (𝚉𝟏𝟏ᵀ𝚉ᵀ)/𝑚
            𝑘  ∼ ∑λₛ⋅ηₛ + ξ                          for 𝑠=𝟷, 𝟸, ..., 𝑆
            ηₛ = χ²(𝟷)

        The terms λₛ are the non-zero eigenvalues of 𝙴ᵀ𝚉ᵀ(𝙸-𝙼)𝚉𝙴. It can also be shown that the
        above (𝑆+2) random variables are pair-wise uncorrelated and that

            𝔼[ξ]   = 𝟎
            𝔼[ξξᵀ] = 𝟺⋅tr[𝙴ᵀ𝚉ᵀ(𝙸-𝙼)𝚉𝙴𝙴ᵀ𝚉ᵀ𝙼𝚉𝙴]

        The p-value of the T statistic is given by:

            P(t<T) = P(min{pᵨ} < T)
                   = 𝟷 - 𝔼[P(𝑘 < min{(2⋅q(pᵨ) - 𝜏ᵨη₀) / ρ} | η₀)],

        where q(pᵨ) is the (𝟷-T)th percentile of the Qᵨ distribution and the expectation is under
        the distribution of η₀. Ideally, we would calculate

            P(t<T) = 1 - ∫F(g(𝑥))⋅p(η₀=𝑥)⋅d𝑥,

        where F(⋅) is the cumulative distribution of 𝑘 and g(𝑥)=min{(2⋅q(pᵨ) - 𝜏ᵨη₀) / ρ}.
        Since we do not know the distribution of ξ, and therefore neither do we know F(⋅), we will
        instead use the cumulative function Fᵪ(⋅) of ∑ηₛ and adjust its mean variance accordingly:

            P(t<T) ≈ 1 - ∫Fᵪ((g(𝑥)-𝜇)⋅c + 𝜇)⋅p(η₀=𝑥)⋅d𝑥,

        where

            𝜇 = 𝔼[𝑘]
            c = √(Var[𝑘] - Var[ξ])/√Var[𝑘].
        """
        # K0 = self._null_lmm_assoc["cov"]
        qscov = self._null_lmm_assoc["qscov"]

        # P = P_matrix(self._W, K0)
        Pmat = PMat(qscov, self._W)
        # H1 vs H0 via score test
        for gr in G.T:
            # D = diag(g)
            g = gr[:, newaxis]

            weights = []
            liu_params = []
            for rho0 in self._rho0:
                # dK = (1 - rho0) * g @ g.T + rho0 * D @ self._EE @ D
                hdK = concatenate(
                    [sqrt(1 - rho0) * g, sqrt(rho0) * ddot(gr, self._E)], axis=1
                )
                ss = ScoreStatistic(Pmat, qscov, hdK)
                Q = ss.statistic(self._y)
                weights += [ss.distr_weights()]
                liu_params += [score_statistic_liu_params(Q, weights)]

            T = min(i["pv"] for i in liu_params)
            q = qmin(liu_params)
            E = self._E

            # 3. Calculate quantities that occur in null distribution
            # g has to be a column-vector
            # D = diag(gr)
            # Pg = P @ g
            Pg = Pmat.dot(g)
            m = (g.T @ Pg)[0, 0]
            # M = 1 / m * (sqrtm(P) @ g @ g.T @ sqrtm(P))
            DE = ddot(gr, E)
            # H1 = E.T @ D.T @ P @ D @ E
            H1 = DE.T @ Pmat.dot(DE)
            # H2 = E.T @ D.T @ sqrtm(P) @ M @ sqrtm(P) @ D @ E
            H2 = 1 / m * multi_dot([DE.T, Pg, Pg.T, ddot(gr, E)])
            H = H1 - H2
            lambdas = eigvalsh(H / 2)

            # eta = ETxPx11xPxE @ ZTIminusMZ

            # Z = sqrtm(P).T @ D
            # I = eye(M.shape[0])
            # eta = E.T @ Z.T @ (I - M) @ Z @ E @ E.T @ Z.T @ M @ Z @ E
            eta_left = (
                ddot(Pmat.dot(DE).T, gr) - multi_dot([DE.T, Pg, ddot(Pg.T, gr)]) / m
            )
            eta_right = multi_dot([E, DE.T, Pg, Pg.T, DE]) / m
            # eta = eta_left @ eta_right
            vareta = 4 * trace2(eta_left, eta_right)

            # OneZTZE = 0.5 * (g.T @ PxoE)
            one = ones((self._n_samples, 1))
            # tau_top = one.T @ Z.T @ Z @ self._E @ self._E.T @ Z.T @ Z @ one
            tau_top = ddot(one.T, gr) @ Pmat.dot(multi_dot([DE, DE.T, Pmat.dot(one)]))
            tau_top = tau_top[0, 0]
            tau_rho = empty(len(self._rho0))
            for i, r0 in enumerate(self._rho0):
                tau_rho[i] = (1 - r0) * m + r0 * tau_top / m

            MuQ = sum(lambdas)
            VarQ = sum(lambdas ** 2) * 2 + vareta
            KerQ = sum(lambdas ** 4) / (sum(lambdas ** 2) ** 2) * 12
            Df = 12 / KerQ

            pvalue = optimal_davies_pvalue(
                q, MuQ, VarQ, KerQ, lambdas, vareta, Df, tau_rho, self._rho0, T
            )
            # Final correction to make sure that the p-value returned is sensible
            # multi = 3
            # if len(self._rhos) < 3:
            #     multi = 2
            # idx = where(pliumod[:, 0] > 0)[0]
            # pval = pliumod[:, 0].min() * multi
            # if pvalue <= 0 or len(idx) < len(self._rhos):
            #     pvalue = pval
            # if pvalue == 0:
            #     if len(idx) > 0:
            #         pvalue = pliumod[:, 0][idx].min()
            return pvalue

    def predict_interaction(self, G):
        G = asarray(G, float)
        Y = self._y[:, newaxis]
        E = self._E
        W = self._W
        n_snps = G.shape[1]
        beta_stars = []
        for i in range(n_snps):
            g = G[:, [i]]
            # mean(𝐲) = W𝛂 + 𝐠𝛽₁ + 𝙴𝝲 = 𝙼𝛃
            M = concatenate((W, g, E), axis=1)
            gE = g * E
            best = {"lml": -inf, "rho1": 0}
            hSigma_p = {}
            for rho1 in self._rho1:
                # Σₚ = ρ₁(𝐠⊙𝙴)(𝐠⊙𝙴)ᵀ + (1-ρ₁)𝙺
                hSigma_p[rho1] = concatenate(
                    (sqrt(rho1) * gE, sqrt(1 - rho1) * self._G), axis=1
                )
                # cov(𝐲) = 𝓋₁Σₚ + 𝓋₂𝙸
                lmm = Kron2Sum(Y, [[1]], M, hSigma_p[rho1], restricted=True)
                lmm.fit(verbose=False)
                if lmm.lml() > best["lml"]:
                    best["lml"] = lmm.lml()
                    best["rho1"] = rho1
                    best["lmm"] = lmm

            lmm = best["lmm"]
            # yadj = 𝐲 - 𝙼𝛃
            yadj = self._y - lmm.mean()
            rho1 = best["rho1"]
            v1 = lmm.C0[0, 0]
            v2 = lmm.C1[0, 0]
            # beta_g = 𝛽₁
            beta_g = lmm.beta[W.shape[1]]
            hSigma_p_qs = economic_qs_linear(hSigma_p[rho1], return_q1=False)
            qscov = QSCov(hSigma_p_qs[0][0], hSigma_p_qs[1], v1, v2)
            # v = cov(𝐲)⁻¹(𝐲 - 𝙼𝛃)
            v = qscov.solve(yadj)
            Estar = vstack([E, E])
            sig2_ge = v1 * rho1
            beta_star = beta_g + sig2_ge * multi_dot([Estar, gE.T, v])
            beta_stars.append(beta_star)
        return asarray(beta_stars, float).T

    def scan_interaction(
        self, G, idx_E: Optional[any] = None, idx_G: Optional[any] = None
    ):
        """
        𝐲 = W𝛂 + 𝐠𝛽₁ + 𝐠⊙𝛃₂ + 𝐞 + 𝐮 + 𝛆
           [fixed=X]   [H1]

        𝛃₂ ~ 𝓝(𝟎, 𝓋₃𝙴𝙴ᵀ),
        𝐞 ~ 𝓝(𝟎, 𝓋₁ρ₁𝙴𝙴ᵀ),
        𝐮 ~ 𝓝(𝟎, 𝓋₁(1-ρ₁)𝙺), and
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
        from time import time

        start = time()
        for i in tqdm(range(n_snps)):
            g = G[:, [i]]
            X = concatenate((self._W, g), axis=1)
            best = {"lml": -inf, "rho1": 0}
            # Null model fitting: find best (𝛂, 𝛽₁, 𝓋₁, 𝓋₂, ρ₁)
            for rho1 in self._rho1:
                # QS = self._Sigma_qs[rho1]
                start = time()
                # halfSigma = self._halfSigma[rho1]
                # Σ = ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺
                # cov(y₀) = 𝓋₁Σ + 𝓋₂I
                QS = self._Sigma_qs[rho1]
                lmm = LMM(self._y, X, QS, restricted=True)
                lmm.fit(verbose=False)
                print(f"Elapsed: {time() - start}")
                print(f"lml: {lmm.lml()}")
                if lmm.lml() > best["lml"]:
                    best["lml"] = lmm.lml()
                    best["rho1"] = rho1
                    best["lmm"] = lmm
                # print(f"Elapsed: {time() - start}")
            # print(f"Elapsed: {time() - start}")
            # print(best["lml"])
            # print(best["rho1"])
            lmm = best["lmm"]
            # H1 via score test
            # Let K₀ = e²𝙴𝙴ᵀ + g²𝙺 + 𝜀²I
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
                E1 = self._E
            else:
                E1 = self._E[idx_E, :]

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

            ss = ScoreStatistic(P, qscov, ddot(gtest, E1))
            Q = ss.statistic(self._y)
            # import numpy as np

            # deltaK = np.diag(gtest) @ EE @ np.diag(gtest)
            # Q_ = 0.5 * self._y.T @ P0 @ deltaK @ P0 @ self._y
            # print(f"Elapsed: {time() - start}")
            # Q is the score statistic for our interaction test and follows a linear combination
            # of chi-squared (df=1) distributions:
            # Q ∼ ∑λχ², where λᵢ are the non-zero eigenvalues of ½√P₀⋅∂K⋅√P₀.
            # Since eigenvals(𝙰𝙰ᵀ) = eigenvals(𝙰ᵀ𝙰) (TODO: find citation),
            # we can compute ½(√∂K)P₀(√∂K) instead.
            # start = time()
            # import scipy as sp
            # sqrtm = sp.linalg.sqrtm
            # np.linalg.eigvalsh(0.5 * sqrtm(P0) @ deltaK @ sqrtm(P0))
            # np.linalg.eigvalsh(0.5 * sqrtm(deltaK) @ P0 @ sqrtm(deltaK))
            # TODO: compare with Liu approximation, maybe try a computational intensive method
            pval, pinfo = davies_pvalue(Q, ss.matrix_for_dist_weights(), True)
            pvalues.append(pval)
            # print(f"Elapsed: {time() - start}")

        info = {key: asarray(v, float) for key, v in info.items()}
        return asarray(pvalues, float), info
