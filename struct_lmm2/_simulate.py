from collections import namedtuple

Variances = namedtuple("Variances", "g gxe k e n")
Simulation = namedtuple(
    "Simulation", "mafs y offset beta_g y_g y_gxe y_k y_e y_n variances G E K"
)


def sample_maf(n_snps: int, maf_min: float, maf_max: float, random):
    assert maf_min <= maf_max and maf_min >= 0 and maf_max <= 1
    return random.rand(n_snps) * (maf_max - maf_min) + maf_min


def sample_genotype(n_samples: int, mafs, random):
    from numpy import asarray, stack

    G = []
    mafs = asarray(mafs, float)
    for maf in mafs:
        probs = [(1 - maf) ** 2, 1 - ((1 - maf) ** 2 + maf ** 2), maf ** 2]
        g = random.choice([0, 1, 2], p=probs, size=n_samples)
        G.append(asarray(g, float))

    return stack(G, axis=1)


def column_normalize(X):
    from numpy import asarray, errstate

    X = asarray(X, float)

    with errstate(divide="raise", invalid="raise"):
        return (X - X.mean(0)) / X.std(0)


def create_environment_matrix(E, n_samples: int, n_rep: int, n_env: int, random):
    """
    The created matrix 𝙴 will represent two environments.
    """
    n = n_samples * n_rep
    rows = random.choice(E.shape[0], n, replace=True)
    cols = random.choice(E.shape[1], n_env, replace=True)
    return E[rows, :][:, cols]


def sample_covariance_matrix(n_samples: int, random, n_rep: int = 1):
    """
    Sample a full-rank covariance matrix.
    """
    from numpy import tile, errstate, eye

    G = random.rand(n_samples, n_samples)
    G = tile(G, (n_rep, 1))
    G = column_normalize(G)
    K = G @ G.T

    with errstate(divide="raise", invalid="raise"):
        # This small diagonal offset is to guarantee the full-rankness.
        K /= K.diagonal().mean() + 1e-8 * eye(n_samples * n_rep)

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
    n_effects: int, causal_indices: list, variance: float, random
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
    from numpy import zeros, errstate, sqrt

    n_causals = len(causal_indices)

    effsizes = zeros(n_effects)
    effsizes[causal_indices] = random.choice([+1, -1], size=len(causal_indices))
    with errstate(divide="raise", invalid="raise"):
        effsizes *= sqrt(variance / n_causals)

    return effsizes


def sample_persistent_effects(G, effsizes, variance: float):
    y_g = G @ effsizes
    _ensure_moments(y_g, 0, variance)
    return y_g


def sample_gxe_effects(G, E, causal_indices: list, variance: float, random):
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
    from numpy import zeros, sqrt

    n_samples = G.shape[0]
    n_envs = E.shape[1]
    n_causals = len(causal_indices)
    vi = variance / n_causals

    y2 = zeros(n_samples)
    for causal in causal_indices:
        # 𝜶ᵢ ∼ 𝓝(𝟎, 𝜎ᵢ²I)
        alpha = sqrt(vi) * random.randn(n_envs)

        # Make the sample statistics close to population
        # statistics
        _ensure_moments(alpha, 0, sqrt(vi))

        # 𝜷 = 𝛜ᵀ𝜶ᵢ
        beta = E @ alpha

        # 𝑔ᵢ⋅𝛜ᵀ𝜶ᵢ
        y2 += G[:, causal] * beta

    _ensure_moments(y2, 0, variance)

    return y2


def sample_environment_effects(E, variance: float, random):
    from numpy import sqrt

    n_envs = E.shape[1]
    effsizes = sqrt(variance) * random.randn(n_envs)
    y3 = E @ effsizes

    _ensure_moments(y3, 0, variance)

    return y3


def sample_population_effects(K, variance: float, random):
    from numpy import zeros

    y4 = random.multivariate_normal(zeros(K.shape[0]), K)

    _ensure_moments(y4, 0, variance)

    return y4


def sample_noise_effects(n_samples: int, variance: float, random):
    from numpy import sqrt

    y5 = sqrt(variance) * random.randn(n_samples)
    _ensure_moments(y5, 0, variance)

    return y5


def sample_phenotype(
    offset: float,
    E,
    n_samples: int,
    n_snps: int,
    n_rep: int,
    n_env: int,
    maf_min: float,
    maf_max: float,
    g_causals: list,
    gxe_causals: list,
    variances: Variances,
    random,
) -> Simulation:
    from numpy import tile

    mafs = sample_maf(n_snps, maf_min, maf_max, random)

    G = sample_genotype(n_samples, mafs, random)
    G = tile(G, (n_rep, 1))
    G = column_normalize(G)
    E = create_environment_matrix(E, n_samples, n_rep, n_env, random)
    #E = tile(E, (n_rep, 1))
    
    #E[n_samples:n_samples*n_rep,:] = -E[n_samples:n_samples*n_rep,:]
    #k = 10
    #n = n_samples*n_rep
    #E = random.randn(n, k)
    E = column_normalize(E)

    K = sample_covariance_matrix(n_samples, random, n_rep)

    beta_g = sample_persistent_effsizes(n_snps, g_causals, variances.g, random)
    y_g = sample_persistent_effects(G, beta_g, variances.g)
    y_gxe = sample_gxe_effects(G, E, gxe_causals, variances.gxe, random)
    y_k = sample_population_effects(K, variances.k, random)
    y_e = sample_environment_effects(E, variances.e, random)
    y_n = sample_noise_effects(n_samples * n_rep, variances.n, random)

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
        K=K,
        E=E,
        G=G,
    )

    return simulation


def _ensure_moments(arr, mean: float, variance: float):
    from numpy import errstate, sqrt

    arr -= arr.mean(0) + mean
    with errstate(divide="raise", invalid="raise"):
        arr /= arr.std(0)
    arr *= sqrt(variance)
