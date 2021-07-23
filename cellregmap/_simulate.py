from collections import namedtuple
from typing import List, Union

from numpy import (
    array_split,
    asarray,
    cumsum,
    errstate,
    eye,
    isscalar,
    ones,
    repeat,
    split,
    sqrt,
    stack,
    zeros,
)
from numpy.random import Generator
from numpy_sugar import ddot
from numpy_sugar.linalg import economic_svd

from ._types import Term

Variances = namedtuple("Variances", "g gxe k e n")
Simulation = namedtuple(
    "Simulation", "mafs y offset beta_g y_g y_gxe y_k y_e y_n variances G E Lk Ls K M"
)
SimulationFixedGxE = namedtuple(
    "Simulation",
    "mafs y offset beta_g beta_gxe beta_e y_g y_gxe y_k y_e y_n variances G E Lk K M",
)


def sample_maf(n_snps: int, maf_min: float, maf_max: float, random: Generator):
    assert maf_min <= maf_max and maf_min >= 0 and maf_max <= 1
    return random.random(n_snps) * (maf_max - maf_min) + maf_min


def sample_genotype(n_samples: int, mafs, random):
    G = []
    mafs = asarray(mafs, float)
    for maf in mafs:
        probs = [(1 - maf) ** 2, 1 - ((1 - maf) ** 2 + maf ** 2), maf ** 2]
        g = random.choice([0, 1, 2], p=probs, size=n_samples)
        G.append(asarray(g, float))

    return stack(G, axis=1)


def column_normalize(X):
    X = asarray(X, float)

    with errstate(divide="raise", invalid="raise"):
        return (X - X.mean(0)) / X.std(0)


def create_environment_matrix(
    n_samples: int, n_env: int, groups: List[List[int]], random: Generator
):
    E = random.normal(size=[n_samples, n_env])
    E = column_normalize(E)
    EE = E @ E.T
    EE /= EE.diagonal().mean()
    H = sample_covariance_matrix(n_samples, groups)[1]
    M = EE + H
    M /= M.diagonal().mean()
    jitter(M)
    return _symmetric_decomp(M)


def create_environment_vector(
    n_samples: int, groups: List[List[int]], random: Generator
):
    E = zeros((n_samples, 1))

    values = random.choice([-1, 1], 2, False)
    for value, group in zip(values, groups):
        E[group, 0] = value

    return E


def sample_covariance_matrix(n_samples: int, groups: List[List[int]]):
    X = zeros((n_samples, len(groups)))

    for i, idx in enumerate(groups):
        X[idx, i] = 1.0

    K = X @ X.T
    K /= K.diagonal().mean()
    jitter(K)

    return (_symmetric_decomp(K), K)


def jitter(K):
    with errstate(divide="raise", invalid="raise"):
        # This small diagonal offset is to guarantee the full-rankness.
        K += 1e-8 * eye(K.shape[0])

    return K


def create_variances(r0, v0, has_kinship=True) -> Variances:
    """
    Remember that:

        cov(𝐲) = 𝓋₀(1-ρ₀)𝙳𝟏𝟏ᵀ𝙳 + 𝓋₀ρ₀𝙳𝙴𝙴ᵀ𝙳 + 𝓋₁ρ₁EEᵀ + 𝓋₁(1-ρ₁)𝙺 + 𝓋₂𝙸.

    Let us define:

        σ²_g   = 𝓋₀(1-ρ₀) (variance explained by persistent genetic effects)
        σ²_gxe = 𝓋₀ρ₀     (variance explained by GxE effects)

        σ²_e   = 𝓋₁ρ₁     (variance explained by environmental effects)
        σ²_k   = 𝓋₁(1-ρ₁) (variance explained by population structure)
        σ²_n   = 𝓋₂       (residual variance, noise)

    We set the total variance to sum up to 1:

        1 = σ²_g + σ²_gxe + σ²_e + σ²_k + σ²_n

    We set the variances explained by the non-genetic terms to be equal:

        v = σ²_e = σ²_k = σ²_n

    For `has_kinship=False`, we instead set the variances such that:

        v = σ²_e = σ²_n

    Parameters
    ----------
    r0 : float
        This is ρ₀.
    v0 : float
        This is 𝓋₀.
    """
    v_g = v0 * (1 - r0)
    v_gxe = v0 * r0

    v_k = 0.0
    if has_kinship:
        v = (1 - v_gxe - v_g) / 3
        v_e = v
        v_k = v
        v_n = v
    else:
        v = (1 - v_gxe - v_g) / 2
        v_e = v
        v_n = v

    variances = {"g": v_g, "gxe": v_gxe, "e": v_e, "n": v_n}
    if has_kinship:
        variances["k"] = v_k
    else:
        variances["k"] = None

    return Variances(**variances)


def sample_persistent_effsizes(
    n_effects: int, causal_indices: list, variance: float, random: Generator
):
    """
    Let 𝚓 denote a sample index and 𝚔 denote a SNP index. Let 𝚟ⱼ = 𝐠ⱼᵀ𝛃.
    We assume that 𝑔ⱼₖ is a random variable such that:

        𝔼[𝑔ⱼₖ] = 0
        𝔼[𝑔ⱼₖ²] = 1

    And we also assume that SNPs are uncorrelated from each other: 𝔼[𝑔ⱼₖ⋅𝑔ⱼᵣ] = 0
    for 𝚔≠𝚛.
    Assuming that 𝛃 is given (fixed), we want to simulate 𝛃 such that:

        𝔼[𝚟ⱼ] = 𝔼[∑ₖ𝑔ⱼₖ𝛽ₖ] = ∑ₖ𝔼[𝑔ⱼₖ]𝛽ₖ = 0
        𝔼[𝚟ⱼ²] = 𝔼[(∑ₖ𝑔ⱼₖ𝛽ₖ)²] = ∑ₖ𝔼[𝑔ⱼₖ²]𝛽ₖ² = ∑ₖ𝛽ₖ² = 𝓋.

    Let 𝚒 denote a causal index. We initialize 𝛃←𝟎 and then randomly set 𝛽ᵢϵ{-1,+1} for
    the causal SNPs. At last, we set 𝛃←𝛃×√(𝓋/𝘯) where 𝘯 is the number of causal SNPs.
    This way we have ∑ₖ𝛽ₖ² = 𝓋.

    Parameters
    ----------
    n_effects : int
        Number of effects.
    causal_indices : list
        List of causal SNPs.
    variance : float
        Correspond to 𝓋.
    """
    n_causals = len(causal_indices)

    effsizes = zeros(n_effects)
    if variance == 0.0:
        return effsizes

    effsizes[causal_indices] = random.choice([+1, -1], size=len(causal_indices))
    with errstate(divide="raise", invalid="raise"):
        effsizes *= sqrt(variance / n_causals)

    return effsizes


def sample_persistent_effects(X, effsizes, variance: float):
    y_g = X @ effsizes
    if variance > 0:
        _ensure_moments(y_g, 0, variance)
    return y_g


def sample_gxe_effects(G, E, causal_indices: list, variance: float, random: Generator):
    """
    Let 𝚒 denote a SNP index and 𝚓 denote an environment.
    Let 𝑦₂ = ∑ᵢ(𝑔ᵢ⋅𝛜ᵀ𝜶ᵢ) be the total GxE effect with

        𝜶ᵢ ∼ 𝓝(𝟎, 𝜎ᵢ²I)

    for every SNP ᵢ.
    We have

        𝔼[𝑦₂] = ∑ᵢ𝔼[𝑔ᵢ⋅𝛜ᵀ𝜶ᵢ] = ∑ᵢ𝔼[𝑔ᵢ]𝔼[𝛜ᵀ𝜶ᵢ] = ∑ᵢ0⋅𝔼[𝛜ᵀ𝜶ᵢ] = 0,

    where 𝑔ᵢ and 𝛜ᵀ𝜶ᵢ are assumed to be uncorrelated.

    We also have

        𝔼[𝑦₂²] = 𝔼[(∑ᵢ𝑔ᵢ⋅𝛜ᵀ𝜶ᵢ)²] = ∑ᵢ∑ⱼ𝔼[𝜖ⱼ²]𝔼[𝛼ᵢⱼ²] = ∑ᵢ𝜎ᵢ² = 𝜎²,

    after a couple of assumptions.

    We define 𝜎ᵢ²=𝑣ᵢ if 𝑔ᵢ is causal and 𝜎ᵢ²=0 otherwise. We assume all causal SNPs
    to have equal effect as defined by 𝑣ᵢ=𝜎²/𝑛₂, where 𝑛₂ is the number of SNPs
    having GxE effects.

    We also assume that 𝔼[𝜖ⱼ]=0 and 𝔼[𝜖ⱼ²]=1/𝑛ₑ for every environment 𝚓.
    """
    n_samples = G.shape[0]
    n_envs = E.shape[1]
    n_causals = len(causal_indices)

    y2 = zeros(n_samples)
    if variance == 0.0:
        return y2

    vi = variance / n_causals
    for causal in causal_indices:
        # 𝜶ᵢ ∼ 𝓝(𝟎, 𝜎ᵢ²I)
        alpha = sqrt(vi) * random.normal(size=n_envs)

        # Make the sample statistics close to population
        # statistics
        if n_envs > 1:
            _ensure_moments(alpha, 0, sqrt(vi))

        # 𝜷 = 𝛜ᵀ𝜶ᵢ
        beta = E @ alpha

        # 𝑔ᵢ⋅𝛜ᵀ𝜶ᵢ
        y2 += G[:, causal] * beta

    _ensure_moments(y2, 0, variance)

    return y2


# def sample_environment_effects(E, variance: float, random):
#     from numpy import sqrt

#     n_envs = E.shape[1]
#     effsizes = sqrt(variance) * random.randn(n_envs)
#     y3 = E @ effsizes

#     _ensure_moments(y3, 0, variance)

#     return y3


# def sample_random_effect(K, variance: float, random: Generator):
#     y = random.multivariate_normal(zeros(K.shape[0]), K, method="cholesky")

#     _ensure_moments(y, 0, variance)

#     return y

def _sample_random_effect(X, variance: float, random: Generator):
    u = sqrt(variance) * random.normal(size=X.shape[1])
    y = X @ u

    _ensure_moments(y, 0, variance)

    return y


def sample_random_effect(X, variance: float, random: Generator):
    if not isinstance(X, tuple):
        return _sample_random_effect(X, variance, random)

    n = X[0].shape[0]
    y = zeros(n)
    for L in X:
        u = sqrt(variance) * random.normal(size=L.shape[1])
        y += L @ u
    _ensure_moments(y, 0, variance)

    return y


def sample_noise_effects(n_samples: int, variance: float, random: Generator):
    y5 = sqrt(variance) * random.normal(size=n_samples)
    _ensure_moments(y5, 0, variance)

    return y5


def sample_phenotype_gxe(
    offset: float,
    n_individuals: int,
    n_snps: int,
    n_cells: Union[int, List[int]],
    n_env_groups: int,
    maf_min: float,
    maf_max: float,
    g_causals: list,
    gxe_causals: list,
    variances: Variances,
    random: Generator,
    env_term: Term = Term.RANDOM,
) -> Simulation:
    """
    Parameters
    ----------
    n_cells
         Integer number of array of integers.
    """
    mafs = sample_maf(n_snps, maf_min, maf_max, random)

    G = sample_genotype(n_individuals, mafs, random)
    G = repeat(G, n_cells, axis=0)
    G = column_normalize(G)

    n_samples = G.shape[0]

    if isscalar(n_cells):
        individual_groups = array_split(range(n_samples), n_individuals)
    else:
        individual_groups = split(range(n_samples), cumsum(n_cells))[:-1]

    env_groups = array_split(random.permutation(range(n_samples)), n_env_groups)

    E = sample_covariance_matrix(n_samples, env_groups)[0]

    Lk, K = sample_covariance_matrix(n_samples, individual_groups)
    [U, S, _] = economic_svd(E)
    us = U * S
    Ls = tuple([ddot(us[:, i], Lk) for i in range(us.shape[1])])

    beta_g = sample_persistent_effsizes(n_snps, g_causals, variances.g, random)
    y_g = sample_persistent_effects(G, beta_g, variances.g)

    y_gxe = sample_gxe_effects(G, E, gxe_causals, variances.gxe, random)

    y_k = sample_random_effect(Ls, variances.k, random)

    if env_term is Term.RANDOM:
        y_e = sample_random_effect(E, variances.e, random)
    elif env_term is Term.FIXED:
        n = E.shape[1]
        beta_e = sample_persistent_effsizes(n, list(range(n)), variances.e, random)
        y_e = sample_persistent_effects(E, beta_e, variances.e)
    else:
        raise ValueError("Invalid term.")

    y_n = sample_noise_effects(n_samples, variances.n, random)

    M = ones((K.shape[0], 1))
    y = offset + y_g + y_gxe + y_k + y_e + y_n

    simulation = Simulation(
        mafs=mafs,
        offset=offset,
        beta_g=beta_g,
        y_g=y_g,
        y_gxe=y_gxe,
        y_k=y_k,
        y_e=y_e,
        y_n=y_n,
        y=y,
        variances=variances,
        Lk=Lk,
        Ls=Ls,
        K=K,
        E=E,
        G=G,
        M=M,
    )

    return simulation


def sample_phenotype(
    offset: float,
    n_individuals: int,
    n_snps: int,
    n_cells: Union[int, List[int]],
    n_env: int,
    n_env_groups: int,
    maf_min: float,
    maf_max: float,
    g_causals: list,
    gxe_causals: list,
    variances: Variances,
    random: Generator,
) -> Simulation:
    """
    Parameters
    ----------
    n_cells
         Integer number of array of integers.
    """
    mafs = sample_maf(n_snps, maf_min, maf_max, random)

    G = sample_genotype(n_individuals, mafs, random)
    G = repeat(G, n_cells, axis=0)
    G = column_normalize(G)

    n_samples = G.shape[0]
    individual_groups = array_split(range(n_samples), n_individuals)

    env_groups = array_split(random.permutation(range(n_samples)), n_env_groups)
    E = create_environment_matrix(n_samples, n_env, env_groups, random)

    Lk, K = sample_covariance_matrix(n_samples, individual_groups)

    beta_g = sample_persistent_effsizes(n_snps, g_causals, variances.g, random)

    y_g = sample_persistent_effects(G, beta_g, variances.g)

    y_gxe = sample_gxe_effects(G, E, gxe_causals, variances.gxe, random)

    y_k = sample_random_effect(Lk, variances.k, random)

    y_e = sample_random_effect(E, variances.e, random)

    y_n = sample_noise_effects(n_samples, variances.n, random)

    M = ones((K.shape[0], 1))
    y = offset + y_g + y_gxe + y_k + y_e + y_n

    simulation = Simulation(
        mafs=mafs,
        offset=offset,
        beta_g=beta_g,
        y_g=y_g,
        y_gxe=y_gxe,
        y_k=y_k,
        y_e=y_e,
        y_n=y_n,
        y=y,
        variances=variances,
        Lk=Lk,
        K=K,
        E=E,
        G=G,
        M=M,
    )

    return simulation


def _ensure_moments(arr, mean: float, variance: float):
    arr -= arr.mean(0) + mean
    with errstate(divide="raise", invalid="raise"):
        arr /= arr.std(0)
    arr *= sqrt(variance)


def _symmetric_decomp(H):
    [U, S, _] = economic_svd(H)
    return ddot(U, sqrt(S))
