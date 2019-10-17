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


def create_environment_matrix(n_samples: int):
    """
    The created matrix 𝙴 will represent two environments.
    """
    from numpy import zeros

    group_size = n_samples // 2
    E = zeros((n_samples, 2))
    E[:group_size, 0] = 1
    E[group_size:, 1] = 1

    return E


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
        K /= K.diagonal().mean() + 1e-4 * eye(n_samples * n_rep)
        K /= K.diagonal().mean()

    return K


def variances(r0, v0, has_kinship=True):
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

    variances = {"v_g": v_g, "v_gxe": v_gxe, "v_e": v_e, "v_n": v_n}
    if has_kinship:
        variances["v_k"] = v_k

    return variances


def sample_persistent_effects(
    n_effects: int, causal_indices: list, variance: float, random
):
    """
    Let ⱼ denote a sample index and ₖ denote a SNP index. Let 𝚟ⱼ = 𝐠ⱼᵀ𝛃.
    We assume that 𝑔ⱼₖ is a random variable such that:

        𝔼[𝑔ⱼₖ] = 0
        𝔼[𝑔ⱼₖ²] = 1

    And we also assume that SNPs are uncorrelated from each other.
    Assuming that 𝛃 is given (fixed), we want to simulate 𝛃 such that:

        𝔼[𝚟ⱼ] = 𝔼[∑ₖ𝑔ⱼₖ𝛽ₖ] = ∑ₖ𝔼[𝑔ⱼₖ]𝛽ₖ = 0
        𝔼[𝚟ⱼ²] = 𝔼[(∑ₖ𝑔ⱼₖ𝛽ₖ)²]

    Using the uncorrelation property between SNPs, we can write

        𝔼[(∑ₖ𝑔ⱼₖ𝛽ₖ)²] = ∑ₖ𝔼[𝑔ⱼₖ²]𝛽ₖ² = ∑ₖ𝛽ₖ² = 𝓋.

    Let ᵢ denote a causal index. We initialize 𝛃=𝟎 and then randomly set 𝛽ᵢϵ{-1,+1} for
    the causal SNPs. At last, we set 𝛃←𝛃×√(𝓋/𝘯) where 𝘯 is the number of causal SNPs so
    that ∑ₖ𝛽ₖ² = 𝓋.
    """
    from numpy import zeros, errstate, sqrt

    n_causals = len(causal_indices)

    effsizes = zeros(n_effects)
    effsizes[causal_indices] = random.choice([+1, -1], size=len(causal_indices))
    with errstate(divide="raise", invalid="raise"):
        effsizes *= sqrt(variance / n_causals)

    # effects =
    with errstate(divide="raise", invalid="raise"):
        effects /= effects.std()
        effects *= variance / len(causal_indices)

    return effects


def sample_gxe_effects(G, E, causal_indices: list, variance: float, random):
    """
    Let ᵢ denote a causal index. We sample 𝐯 = ∑ᵢ𝐠ᵢ⊙𝛃ᵢ such that:

        𝛃ᵢ ∼ 𝓝(𝟎, 𝓋ᵢ𝙴𝙴ᵀ)

    and 𝔼[𝐯ᵀ𝐯] = 𝓋 and 𝓋ᵢ = 𝓋 / 𝘯, for 𝘯 being the number of causal SNPs.
    """
    from numpy import zeros, errstate, sqrt

    n_samples = G.shape[0]
    n_envs = E.shape[1]
    n_causals = len(causal_indices)
    vi = variance / n_causals

    v = zeros(n_samples)
    for causal in causal_indices:
        # Let 𝐮 ∼ 𝓝(𝟎, 𝙸) and 𝛃 = σ𝙴𝐮.
        # We have 𝔼[𝛃] = σ𝙴𝔼[𝐮]= 𝟎 and 𝔼[𝛃ᵀ𝛃] = 𝔼[σ𝙴𝐮𝐮ᵀ𝙴ᵀσ] = σ²𝙴𝔼[𝐮𝐮ᵀ]𝙴ᵀ =
        # Therefore, 𝛃 ∼ 𝓝(𝟎, σ²𝙴𝙴ᵀ).
        u = random.randn(n_envs)
        beta = sqrt(vi) * (E @ u)
        eff = G[:, causal] * beta
        with errstate(divide="raise", invalid="raise"):
            eff /= eff.std(0)
        eff *= vi
        v += eff

    v -= v.mean(0)
    with errstate(divide="raise", invalid="raise"):
        v /= v.std(0)
    v *= variance

    return v


def sample_phenotype():
    pass
