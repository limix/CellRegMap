## Relation to LIMIX

CellRegMap's linear mixed model (LMM) uses the FaST-LMM implementation described [here](https://www.nature.com/articles/nmeth.1681) and used within the [LIMIX](https://github.com/limix/limix) framework.

### Linear Mixed Model implementation using LIMIX
LIMIX is described in [this preprint](https://www.biorxiv.org/content/10.1101/003905v2) and documentation can be found [here](https://limix-tempdoc.readthedocs.io/en/latest/).
While there are several fast implementations to map eQTLs using correlation or linear regression-based approaches (e.g., [tensorQTL](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1836-7), [matrix eQTL](https://academic.oup.com/bioinformatics/article/28/10/1353/213326?login=true)), to our knowledge LIMIX is the fastest software out there for genetic analyses using **linear mixed models** (which allow to better model population stratification and cryptic relatedness, which are prevalent in human genetic data - see for example thread [here](https://twitter.com/shaicarmi/status/1508298704796663808?s=21&t=6xaF5BmozHil3VbXotlGhQ)).

### eQTL mapping using LIMIX
Additionally, for a LIMIX wrapper specifically to map eQTL, see our [limix QTL pipeline](https://github.com/single-cell-genetics/limix_qtl) which we most recently used in our [Genome Biology Publication](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02407-x).

The pipeline provides an easy wrapper to map eQTLs using various methods, automatically re-ordering, subselecting and expanding files to match with each other.
See the wiki pages for [installation](https://github.com/single-cell-genetics/limix_qtl/wiki/Installation) and [input files](https://github.com/single-cell-genetics/limix_qtl/wiki/Inputs), and an example [snakemake for standard eQTL mapping]().

### Coming Soon
With [Marc Jan Bonder](https://twitter.com/mjbonder), we are in the process of implementing CellRegMap runners compatible with this pipeline.

Also see our sister project [scDALI](https://pmbio.github.io/scdali/), a model for modelling allelic imbalance in single cells.

<!-- For standard eQTL mapping within a homogeneous population of single cells, pseudo-bulk and bulk-like approaches as described in the GB paper can be used.
We recommend using CellRegMap in the presence of more continuous cellular states, or rarer cell types.
In those scenarios, modelling the full transcriptome across donors and states can improve power. -->
