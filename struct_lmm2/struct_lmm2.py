class StructLMM2:
    r"""
    Mixed-model with genetic effect heterogeneity.

    The extended StructLMM model (two random effects) is:

        𝐲 = W𝛂 + 𝐠⊙𝛃 + 𝐞 + 𝐮 + 𝛆,                                               (1)

    where:

        𝛃 ∼ 𝓝(𝟎, 𝓋₀((1-ρ₀)𝟏𝟏ᵀ + ρ₀𝙴𝙴ᵀ)),
        𝐞 ∼ 𝓝(𝟎, 𝓋₁ρ₁EEᵀ),
        𝐮 ~ 𝓝(𝟎, 𝓋₁(1-ρ₁)𝙺), and
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
        pass
