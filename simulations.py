from numpy import asarray, eye, sqrt, stack, zeros
from numpy.random import RandomState

# Let Σ = 𝙴𝙴ᵀ
# 𝐲 ∼ 𝓝(𝙼𝛂, 𝓋₀𝙳(ρ𝟏𝟏ᵀ + (1-ρ)Σ)𝙳 + 𝓋₁(aΣ + (1-a)𝙺) + 𝓋₂𝙸).

breakpoint()
seed = 0
random = RandomState(seed)
n_samples = 100
# simulate MAF (minor allele frequency) distribution
maf_min = 0.1
maf_max = 0.45
n_snps = 50

# two groups
group_size = n_samples // 2

E = zeros((n_samples, 2))
E[:group_size, 0] = 1
E[group_size:, 1] = 1

Sigma = E @ E.T

# Simulate genotypes (for n_snps variants)
mafs = random.rand(n_snps) * (maf_max - maf_min) + maf_min

# simulate SNPs accordingly
G = []

for maf in mafs:
    g = random.choice(
        [0, 1, 2],
        p=[(1 - maf) ** 2, 1 - ((1 - maf) ** 2 + maf ** 2), maf ** 2],
        size=n_samples,
    )
    G.append(asarray(g, float))

# We normalize it such that the expectation of 𝔼[𝐠ᵀ𝐠] = 1.
G = stack(G, axis=1)
G -= G.mean(0)
G /= G.std(0)
G /= sqrt(G.shape[1])
K = G @ G.T

idxs_persistent = [5, 30]
idxs_gxe = [30, 45]

# Variances
#
# 𝐲 ∼ 𝓝(1 + ∑ᵢ𝐠ᵢ𝛽_gᵢ, σ²_g⋅𝙳𝟏𝟏ᵀ𝙳 + σ²_gxe⋅𝙳Σ𝙳𝙳 + σ²_e⋅Σ + σ²_k⋅𝙺 + σ²_n⋅𝙸.
# σ²_g + σ²_gxe + σ²_e + σ²_k + σ²_n = 1
# σ²₁*ρ + σ²₁*(1-ρ) + σ²_e + σ²_k + σ²_n = 1

# The user will provide: σ²₁, ρ
# And we assume that σ²_e = σ²_k = σ²_n = v
# v = (1 - σ²₁*ρ + σ²₁*(1-ρ)) / 3
# σ²_e = a*σ²₂
# σ²_k = (1-a)*σ²₂

# 'simulate sigma parameters'
rho = 0.2
var_tot_g_gxe = 0.3
var_tot_g = rho * var_tot_g_gxe
var_tot_gxe = (1 - rho) * var_tot_g_gxe

var_g = var_tot_g / len(idxs_persistent)
var_gxe = var_tot_gxe / len(idxs_gxe)

v = (1 - var_tot_gxe - var_tot_g) / 3
var_e = v  # environment effect only
var_k = v  # population structure effect ?
var_noise = v

# (persistent) genotype portion of phenotype:
#
#     𝐲_g = ∑ᵢ𝐠ᵢ𝛽_gᵢ,
#
# where 𝐠ᵢ is the i-th column of 𝙶.
#
# simulate (persistent) beta to have causal SNPs as defined
beta_g = zeros(n_snps)
beta_g[idxs_persistent] = random.choice([+1, -1], size=len(idxs_persistent))
beta_g /= beta_g.std()
beta_g *= sqrt(var_tot_g)
breakpoint()
# calculate genoytpe component of y
y_g = G @ beta_g

# GxE portion of phenotype:
#
# 	𝐲_gxe = ∑ᵢ gᵢ x 𝛃ᵢ

# simulate (GxE) variance component to have causal SNPs as defined
sigma_gxe = zeros(n_snps)
sigma_gxe[idxs_gxe] = var_gxe

y_gxe = zeros(n_samples)

for i in range(n_snps):
    beta_gxe = random.multivariate_normal(zeros(n_samples), sigma_gxe[i] * Sigma)
    y_gxe += G[:, i] * beta_gxe

breakpoint()
e = random.multivariate_normal(zeros(n_samples), v * Sigma)
u = random.multivariate_normal(zeros(n_samples), v * K)
eps = random.multivariate_normal(zeros(n_samples), v * eye(n_samples))

# 'sum all parts of y'

y = 1 + y_g + y_gxe + e + u + eps
