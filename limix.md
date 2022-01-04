# Relation to LIMIX

CellRegMap's linear mixed model (LMM) uses the FaST-LMM implementation described [here](https://www.nature.com/articles/nmeth.1681) and used within the [limix](https://github.com/limix/limix) framework.

## Linear Mixed Model implementation using LIMIX
limix is described in [this preprint](https://www.biorxiv.org/content/10.1101/003905v2) and documentation can be found [here](https://limix-tempdoc.readthedocs.io/en/latest/).
While there are several fast implementations to map eQTL using correlation or linear regression-based approaches (e.g., [tensorQTL](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1836-7)), to our knowledge limix is the fastest software out there for genetic analyses using linear mixed models.

## eQTL mapping using LIMIX
Additionally, for a limix wrapper specifically to map eQTL, see our [limix QTL pipeline](https://github.com/single-cell-genetics/limix_qtl) which we used in our recent [Genome Biology Publication](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02407-x).

For standard eQTL mapping within a homogeneous population of single cells, pseudo-bulk and bulk-like approaches as described in the paper can be used.
We recommend using CellRegMap in the presence of more continuous cellular states, or rarer cell types.
In those scenarios, modelling the full transcriptome across donors and states can improve power.
