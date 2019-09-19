from glimix_core.lmm import LMM
from numpy import concatenate, empty, inf, ones, sqrt, sqrtm, stack
from numpy.linalg import eigvalsh, inv, solve
from numpy_sugar import ddot
from numpy_sugar.linalg import economic_qs


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

        cov(𝐲) = 𝓋₀(1-ρ₀)𝟏𝟏ᵀ + 𝓋₀ρ₀𝙴𝙴ᵀ + 𝓋₁ρ₁EEᵀ + 𝓋₁(1-ρ₁)𝙺 + 𝓋₂𝙸.

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
            self._Sigma[rho1] = rho1 * self._EE + (1 - rho1) * self.K
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
        alpha = lmm.beta[:-1]
        beta = lmm.beta[-1]
        # e²Σ + g²K = v1(rho1*EEt + (1-rho1)*K)
        # e² = v1*rho1
        # g² = v1*(1-rho1)
        s2 = lmm.v0  # s²
        eps2 = lmm.v1  # 𝜀²
        # Let K₀ = g²K + e²EEt + 𝜀²I
        K0 = lmm.covariance()

        X = self._W

        # Let P₀ = K₀⁻¹ - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹.
        # K0iX = solve(K0, X)
        P0 = inv(K0) - K0iX @ solve(X.T @ K0iX, K0iX.T)

        n_samples = self._y.shape[0]

        nE = self._E.shape[1] + 1
        F = empty((nE, nE))

        # H1 vs H0 via score test
        for i in range(n_snps):
            g = G[:, i].reshape(G.shape[0], 1)

            # # The covariance matrix of H1 is K = K₀ + b²diag(𝐠)⋅Σ⋅diag(𝐠)
            # # We have ∂K/∂b² = diag(𝐠)⋅Σ⋅diag(𝐠)
            # # The score test statistics is given by
            # # Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
            # dK_G = ddot(g.ravel(), ddot(ones((n_samples, n_samples)), g.ravel()))
            # dK_GxE = ddot(g.ravel(), ddot(self._EE, g.ravel()))

            # # P0 = P0 + 1e-9 * eye(P0.shape[0])
            # Q_G = P0y.T @ dK_G @ P0y
            # Q_GxE = P0y.T @ dK_GxE @ P0y

            # # the eigenvalues of ½P₀⋅∂K⋅P₀
            # # are tge eigenvalues of
            # gPg = g.T @ P0 @ g
            # goE = g * self._E
            # gPgoE = g.T @ P0 @ goE
            # gEPgE = goE.T @ P0 @ goE

            # lambdas = []
            # Q = []
            # for rho0 in self._rho0:
            #     Q.append((rho0 * Q_G + (1 - rho0) * Q_GxE) / 2)
            #     F[0, 0] = rho0 * gPg
            #     F[0, 1:] = sqrt(rho0) * sqrt(1 - rho0) * gPgoE
            #     F[1:, 0] = F[0, 1:]
            #     F[1:, 1:] = (1 - rho0) * gEPgE
            #     lambdas.append(eigvalsh(F) / 2)

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

    def _score_stats_null_dist(self, g):
        """
        Under the null hypothesis, the score-based test statistic follows a weighted sum
        of random variables:
            𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1),
        where 𝜆ᵢ are the non-zero eigenvalues of ½√𝙿(∂𝙺)√𝙿.
        Note that
            ∂𝙺ᵨ = 𝙳(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)𝙳 = (ρ𝐠𝐠ᵀ + (1-ρ)𝙴̃𝙴̃ᵀ)
        for 𝙴̃ = 𝙳𝙴.
        By using SVD decomposition, one can show that the non-zero eigenvalues of 𝚇𝚇ᵀ
        are equal to the non-zero eigenvalues of 𝚇ᵀ𝚇.
        Therefore, 𝜆ᵢ are the non-zero eigenvalues of
            ½[√ρ𝐠 √(1-ρ)𝙴̃]𝙿[√ρ𝐠 √(1-ρ)𝙴̃]ᵀ.
        """
        # P₀𝐲 = K⁻¹𝐲 - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹𝐲.

        K0 = self._null_lmm_assoc["lmm"].covariance()
        K0iy = solve(K0, self._y)
        X = self._W
        P0y = K0iy - solve(K0, X @ solve(X.T @ K0iX, X.T @ K0iy))

        # The covariance matrix of H1 is K = K₀ + b²diag(𝐠)⋅Σ⋅diag(𝐠)
        # We have ∂K/∂b² = diag(𝐠)⋅Σ⋅diag(𝐠)
        # The score test statistics is given by
        # Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
        n_samples = len(g)
        dK_G = ddot(g.ravel(), ddot(ones((n_samples, n_samples)), g.ravel()))
        dK_GxE = ddot(g.ravel(), ddot(self._EE, g.ravel()))

        # P0 = P0 + 1e-9 * eye(P0.shape[0])
        Q_G = P0y.T @ dK_G @ P0y
        Q_GxE = P0y.T @ dK_GxE @ P0y

        P0 = inv(K0) - K0iX @ solve(X.T @ K0iX, K0iX.T)
        # the eigenvalues of ½P₀⋅∂K⋅P₀
        # are tge eigenvalues of
        gPg = g.T @ P0 @ g
        goE = g * self._E
        gPgoE = g.T @ P0 @ goE
        gEPgE = goE.T @ P0 @ goE

        lambdas = []
        Q = []
        for rho0 in self._rho0:
            Q.append((rho0 * Q_G + (1 - rho0) * Q_GxE) / 2)
            F[0, 0] = rho0 * gPg
            F[0, 1:] = sqrt(rho0) * sqrt(1 - rho0) * gPgoE
            F[1:, 0] = F[0, 1:]
            F[1:, 1:] = (1 - rho0) * gEPgE
            lambdas.append(eigvalsh(F) / 2)

        return lambdas


def _score_stats_null_dist(g):
    """
    Under the null hypothesis, the score-based test statistic follows a weighted sum
    of random variables:
        𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1),
    where 𝜆ᵢ are the non-zero eigenvalues of ½√𝙿(∂𝙺)√𝙿.
    Note that
        ∂𝙺ᵨ = 𝙳(ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ)𝙳 = (ρ𝐠𝐠ᵀ + (1-ρ)𝙴̃𝙴̃ᵀ)
    for 𝙴̃ = 𝙳𝙴.
    By using SVD decomposition, one can show that the non-zero eigenvalues of 𝚇𝚇ᵀ
    are equal to the non-zero eigenvalues of 𝚇ᵀ𝚇.
    Therefore, 𝜆ᵢ are the non-zero eigenvalues of
        ½[√ρ𝐠 √(1-ρ)𝙴̃]𝙿[√ρ𝐠 √(1-ρ)𝙴̃]ᵀ.
    """
    # P₀𝐲 = K⁻¹𝐲 - K₀⁻¹X(XᵀK₀⁻¹X)⁻¹XᵀK₀⁻¹𝐲.
    K0 = self._null_lmm_assoc["lmm"].covariance()
    K0iy = solve(K0, self._y)
    X = self._W
    K0iX = solve(K0, X)
    P0y = K0iy - solve(K0, X @ solve(X.T @ K0iX, X.T @ K0iy))

    # The covariance matrix of H1 is K = K₀ + b²diag(𝐠)⋅Σ⋅diag(𝐠)
    # We have ∂K/∂b² = diag(𝐠)⋅Σ⋅diag(𝐠)
    # The score test statistics is given by
    # Q = ½𝐲ᵀP₀⋅∂K⋅P₀𝐲
    n_samples = len(g)
    dK_G = ddot(g.ravel(), ddot(ones((n_samples, n_samples)), g.ravel()))
    dK_GxE = ddot(g.ravel(), ddot(self._EE, g.ravel()))

    # P0 = P0 + 1e-9 * eye(P0.shape[0])
    Q_G = P0y.T @ dK_G @ P0y
    Q_GxE = P0y.T @ dK_GxE @ P0y

    P0 = inv(K0) - K0iX @ solve(X.T @ K0iX, K0iX.T)
    # the eigenvalues of ½P₀⋅∂K⋅P₀
    # are tge eigenvalues of
    gPg = g.T @ P0 @ g
    goE = g * self._E
    gPgoE = g.T @ P0 @ goE
    gEPgE = goE.T @ P0 @ goE

    lambdas = []
    Q = []
    for rho0 in self._rho0:
        Q.append((rho0 * Q_G + (1 - rho0) * Q_GxE) / 2)
        F[0, 0] = rho0 * gPg
        F[0, 1:] = sqrt(rho0) * sqrt(1 - rho0) * gPgoE
        F[1:, 0] = F[0, 1:]
        F[1:, 1:] = (1 - rho0) * gEPgE
        lambdas.append(eigvalsh(F) / 2)

    return lambdas


def compute_P(X, K):
    """
    Let 𝐲 ∼ 𝓝(X, 𝙺). It computes 𝙿 = 𝙺⁻¹ - 𝙺⁻¹X(Xᵀ𝙺⁻¹X)⁻¹Xᵀ𝙺⁻¹.
    """
    KiX = solve(K, X)
    return inv(K) - KiX @ solve(X.T @ KiX, KiX.T)


def score_statistic(X, K, dK):
    """
    Let 𝐲 ∼ 𝓝(X, 𝙺). We employ the score-test statistic:

        𝑄 = ½𝐲ᵀ𝙿(∂𝙺)𝙿𝐲,

    where:

        𝙿 = 𝙺⁻¹ - 𝙺⁻¹X(Xᵀ𝙺⁻¹X)⁻¹Xᵀ𝙺⁻¹.

    The score-test statistic follows a weighted sum of random variables:

        𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1),

    where 𝜆ᵢ are the non-zero eigenvalues of ½√𝙿(∂𝙺)√𝙿.
    """
    P = compute_P(X, K)
    return eigvalsh(sqrtm(P) @ dK @ sqrtm(P)) / 2


def score_statistic_params_liu(q, lambdas):
    """
    Computes Pr(𝑄 > q) for 𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1)
    using the Liu survival function approximation.
    [REF]
    """
    from chiscore import liu_sf

    n = len(lambdas)
    # We use the Liu survival function to approximate the distribution followed by a
    # linear combination of noncentral chi-squared variables (Q) using only three parameters
    # of such distribution: the weights, degrees of freedom, and noncentrality (Qh).
    #   𝑄 ∼ ∑λᵢχ²(hᵢ, 𝛿ᵢ),
    # where λᵢ, hᵢ, and 𝛿ᵢ are the weights, degrees of freedom (1), and noncentrality (0)
    # parameters. By setting the last input to True we use the better modified version [REF].
    (pv, dof_x, _, info) = liu_sf(q, lambdas, [1] * len(w), [0] * len(w), True)
    return (pv, info["mu_q"], info["sigma_q"], dof_x)


def _Qmin(liu_params):
    from numpy import zeros
    import scipy.stats as st

    n = len(liu_params)

    # T statistic
    T = min(i["pv"] for i in liu_params)

    qmin = zeros(n)
    percentile = 1 - T
    for i in range(n):
        q = st.chi2.ppf(percentile, liu_params[i]["dof"])
        mu_q = liu_params[i]["mu_q"]
        sigma_q = liu_params[i]["sigma_q"]
        dof = liu_params[i]["dof"]
        qmin[i] = (q - dof) / (2 * dof) ** 0.5 * sigma_q + mu_q

    return qmin
