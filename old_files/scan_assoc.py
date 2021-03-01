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
