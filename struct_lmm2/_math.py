from numpy.linalg import eigvalsh, inv, solve
from scipy.linalg import sqrtm


def P_matrix(X, K):
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
    P = P_matrix(X, K)
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
    (pv, dof_x, _, info) = liu_sf(q, lambdas, [1] * n, [0] * n, True)
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
