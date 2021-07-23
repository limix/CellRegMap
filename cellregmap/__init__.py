from ._types import Term
from ._struct_lmm2 import StructLMM2
from ._simulate import (
    sample_phenotype,
    sample_phenotype_gxe,
    create_variances,
)

__version__ = "0.0.2"

__all__ = [
    "__version__",
    "StructLMM2",
    "sample_phenotype",
    "create_variances",
    "sample_phenotype_gxe",
    "Term",
]
