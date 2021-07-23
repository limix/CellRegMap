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
from numpy import finfo, logical_not, sqrt
from numpy.linalg import eigh, eigvalsh, inv, lstsq, solve, svd
from numpy_sugar import ddot
from scipy.linalg import sqrtm


def rsolve(a, b):
    """
    Robust solver.
    """
    return lstsq(a, b, rcond=None)[0]


class QSCov:
    """
    Represents 𝑎𝙺 + 𝑏𝙸.

    𝙺 matrix is defined by its eigen decomposition, `QS`.
    """

    def __init__(self, Q0, S0, a=1.0, b=1.0):
        self._Q0 = Q0
        self._S0 = S0
        self._a = a
        self._b = b

    def dot(self, v):
        left = self._a * self._Q0 @ ddot(self._S0, self._Q0.T @ v, left=True)
        right = self._b * v
        return left + right

    def solve(self, v):
        # nrows = self._Q0.shape[0]
        # rank = self._Q0.shape[1]

        # tmp = ones(nrows)
        # tmp[:rank] += (self._a / self._b) * self._S0
        # R = 1 / tmp

        # R0 = R[:rank]
        # R1 = R[rank:]  # Always ones

        # tmp = (self._a / self._b) * self._S0
        R0 = 1 / (1 + ((self._a / self._b) * self._S0))
        # R0 = R[:rank]
        Q0v = self._Q0.T @ v
        return (self._Q0 @ ddot(R0, Q0v, left=True) + v - self._Q0 @ Q0v) / self._b
        # left = self._Q0 @ ddot(R0, self._Q0.T @ v, left=True)
        # right = self._Q1 @ ddot(R1, self._Q1.T @ v, left=True)
        # return (left + right) / self._b


class PMat:
    """
    Represents 𝙿 = 𝙺⁻¹ - 𝙺⁻¹𝚆(𝚆ᵀ𝙺⁻¹𝚆)⁻¹𝚆ᵀ𝙺⁻¹.

    The 𝙺 is defined via an `QSCov` object.
    """

    def __init__(self, qscov: QSCov, W):
        self._qscov = qscov
        self._W = W
        self._KiW = self._qscov.solve(self._W)

    def dot(self, v):
        Kiv = self._qscov.solve(v)
        return Kiv - self._KiW @ rsolve(self._W.T @ self._KiW, self._KiW.T @ v)


def P_matrix(W, K):
    """ Computes 𝙿 = 𝙺⁻¹ - 𝙺⁻¹𝚆(𝚆ᵀ𝙺⁻¹𝚆)⁻¹𝚆ᵀ𝙺⁻¹. """
    KiW = solve(K, W)
    return inv(K) - KiW @ solve(W.T @ KiW, KiW.T)


class ScoreStatistic:
    """
    Score-test statistic [1]_ is given by

        𝑄 = ½𝐲ᵀ𝙿(∂𝙺)𝙿𝐲.
    """

    def __init__(self, P: PMat, K: QSCov, sqrt_dK):
        self._P = P
        self._K = K
        self._sqrt_dK = sqrt_dK

    def statistic(self, y):
        """ Compute 𝑄 = ½𝐲ᵀ𝙿(∂𝙺)𝙿𝐲. """
        Py = self._P.dot(y)
        return Py.T @ self._sqrt_dK @ self._sqrt_dK.T @ Py / 2

    def matrix_for_dist_weights(self):
        """Compute ½(√∂𝙺)𝙿(√∂𝙺).

        The returned matrix has its eigenvalues equal to the eigenvalues of ½√𝙿(∂𝙺)√𝙿.
        """
        return self._sqrt_dK.T @ self._P.dot(self._sqrt_dK) / 2

    def distr_weights(self):
        weights = eigvalsh(self.matrix_for_dist_weights())
        return weights[weights > 1e-16]


def score_statistic(y, W, K, dK):
    """
    Score-test statistic [1]_ is given by

        𝑄 = ½𝐲ᵀ𝙿(∂𝙺)𝙿𝐲.
    """
    P = P_matrix(W, K)
    return y.T @ P @ dK @ P @ y / 2


def score_statistic_qs(y, W, qscov, dK):
    """
    Same as `score_statistic`.
    """
    P = PMat(qscov, W)
    Py = P.dot(y)
    return Py.T @ dK @ Py / 2


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
    # linear combination of noncentral chi-squared variables (Q) using only three
    # parameters  of such distribution: the weights, degrees of freedom, and
    # noncentrality (Qh).  𝑄 ∼ ∑λᵢχ²(hᵢ, 𝛿ᵢ),
    # where λᵢ, hᵢ, and 𝛿ᵢ are the weights, degrees of freedom (1), and noncentral
    # (0) parameters. By setting the last input to True we use the modified version
    # [REF].
    (pv, dof_x, _, info) = liu_sf(q, weights, [1] * n, [0] * n, True)
    return {"pv": pv, "mu_q": info["mu_q"], "sigma_q": info["sigma_q"], "dof_x": dof_x}


def qmin(liu_params):
    import scipy.stats as st
    from numpy import zeros

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


def economic_qs(K, epsilon=sqrt(finfo(float).eps)):
    r"""Economic eigen decomposition for symmetric matrices.

    A symmetric matrix ``K`` can be decomposed in
    :math:`\mathrm Q_0 \mathrm S_0 \mathrm Q_0^\intercal + \mathrm Q_1\
    \mathrm S_1 \mathrm Q_1^ \intercal`, where :math:`\mathrm S_1` is a zero
    matrix with size determined by ``K``'s rank deficiency.

    Args:
        K (array_like): Symmetric matrix.
        epsilon (float): Eigen value threshold. Default is
                         ``sqrt(finfo(float).eps)``.

    Returns:
        tuple: ``((Q0, Q1), S0)``.
    """

    (S, Q) = eigh(K)

    nok = abs(max(Q[0].min(), Q[0].max(), key=abs)) < epsilon
    nok = nok and abs(max(K.min(), K.max(), key=abs)) >= epsilon
    if nok:
        from scipy.linalg import eigh as sp_eigh

        (S, Q) = sp_eigh(K)

    ok = S >= epsilon
    nok = logical_not(ok)
    S0 = S[ok]
    Q0 = Q[:, ok]
    Q1 = Q[:, nok]
    return ((Q0, Q1), S0)


def economic_qs_linear(G):
    r"""Economic eigen decomposition for symmetric matrices ``dot(G, G.T)``.

    It is theoretically equivalent to ``economic_qs(dot(G, G.T))``.
    Refer to :func:`numpy_sugar.economic_qs` for further information.

    Args:
        G (array_like): Matrix.

    Returns:
        tuple: ``((Q0, Q1), S0)``.
    """
    if G.shape[0] > G.shape[1]:
        (Q0, Ssq, _) = svd(G, full_matrices=False)
        S0 = Ssq ** 2
        return (Q0, S0)

    Q, S0 = economic_qs(G.dot(G.T))
    return (Q[0], S0)
