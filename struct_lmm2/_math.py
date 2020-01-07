"""
Mathematical definitions
------------------------

This module defines some core mathematical concepts on which
StructLMM2 depends on.
The implementations provided here are meant to help understand
those concepts and to provide test cases.

We assume the definition

    𝐲 ∼ 𝓝(𝚆, 𝙺)

throughout this module.

References
----------
.. [1] Lippert, C., Xiang, J., Horta, D., Widmer, C., Kadie, C., Heckerman, D.,
   & Listgarten, J. (2014). Greater power and computational efficiency for
   kernel-based association testing of sets of genetic variants. Bioinformatics,
   30(22), 3206-3214.
.. [2] Liu, H., Tang, Y., & Zhang, H. H. (2009). A new chi-square approximation
   to the distribution of non-negative definite quadratic forms in non-central
   normal variables. Computational Statistics & Data Analysis, 53(4), 853-856.
.. [3] Lee, Seunggeun, Michael C. Wu, and Xihong Lin. "Optimal tests for rare
   variant effects in sequencing association studies." Biostatistics 13.4 (2012):
   762-775.
"""
from numpy import concatenate, ones
from numpy.linalg import eigvalsh, inv, lstsq, solve
from numpy_sugar import ddot
from scipy.linalg import sqrtm


def rsolve(a, b):
    """
    Robust solver.
    """
    return lstsq(a, b, rcond=None)[0]


class QSCov:
    def __init__(self, QS, a=1.0, b=1.0):
        self._Q0 = QS[0][0]
        self._Q1 = QS[0][1]
        self._S0 = QS[1]
        self._a = a
        self._b = b

    def _Qt_dot(self, v):
        return concatenate([self._Q0.T @ v, self._Q1.T @ v], axis=0)

    def dot(self, v):
        left = self._a * self._Q0 @ ddot(self._S0, self._Q0.T @ v, left=True)
        right = self._b * v
        return left + right

    def solve(self, v):
        SI = ones(self._Q0.shape[0])
        SI[: self._S0.shape[0]] += (self._a / self._b) * self._S0
        return self._Qt_dot(ddot(1 / SI, self._Qt_dot(v))) / self._b


def P_matrix(W, K):
    """ Computes 𝙿 = 𝙺⁻¹ - 𝙺⁻¹𝚆(𝚆ᵀ𝙺⁻¹𝚆)⁻¹𝚆ᵀ𝙺⁻¹. """
    KiX = solve(K, W)
    return inv(K) - KiX @ solve(W.T @ KiX, KiX.T)


def score_statistic(y, W, K, dK):
    """
    Score-test statistic [1]_ is given by

        𝑄 = ½𝐲ᵀ𝙿(∂𝙺)𝙿𝐲.
    """
    P = P_matrix(W, K)
    return y.T @ P @ dK @ P @ y / 2


def score_statistic_distr_weights(W, K, dK):
    """
    Score-test statistic follows a weighted sum of random variables [1]_:

        𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1),

    where 𝜆ᵢ are the non-zero eigenvalues of ½√𝙿(∂𝙺)√𝙿.
    """
    P = P_matrix(W, K)
    weights = eigvalsh(sqrtm(P) @ dK @ sqrtm(P)) / 2
    return weights[weights > 1e-16]


def score_statistic_liu_params(q, weights):
    """
    Computes Pr(𝑄 > q) for 𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1) using a modification [3]_ of the original Liu
    survival function approximation [2]_. This function also returns estimated
    parameters, not yet fully explained here.
    """
    from chiscore import liu_sf

    n = len(weights)
    # We use the Liu survival function to approximate the distribution followed by a
    # linear combination of noncentral chi-squared variables (Q) using only three parameters
    # of such distribution: the weights, degrees of freedom, and noncentrality (Qh).
    #   𝑄 ∼ ∑λᵢχ²(hᵢ, 𝛿ᵢ),
    # where λᵢ, hᵢ, and 𝛿ᵢ are the weights, degrees of freedom (1), and noncentrality (0)
    # parameters. By setting the last input to True we use the modified version [REF].
    (pv, dof_x, _, info) = liu_sf(q, weights, [1] * n, [0] * n, True)
    return {"pv": pv, "mu_q": info["mu_q"], "sigma_q": info["sigma_q"], "dof_x": dof_x}


def qmin(liu_params):
    from numpy import zeros
    import scipy.stats as st

    n = len(liu_params)

    # T statistic
    T = min(i["pv"] for i in liu_params)

    qmin = zeros(n)
    percentile = 1 - T
    for i in range(n):
        q = st.chi2.ppf(percentile, liu_params[i]["dof_x"])
        mu_q = liu_params[i]["mu_q"]
        sigma_q = liu_params[i]["sigma_q"]
        dof = liu_params[i]["dof_x"]
        qmin[i] = (q - dof) / (2 * dof) ** 0.5 * sigma_q + mu_q

    return qmin
