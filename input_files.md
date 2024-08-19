---
layout: default
title: "Input Files"
mathjax: true
---

# The CellRegMap model

The CellRegMap model can be cast as:

$y = W\alpha %2B g\beta_G %2B g \odot \beta_{GxC} %2B c %2B u %2B \epsilon$

<img src="https://render.githubusercontent.com/render/math?math=y = W\alpha %2B g\beta_G %2B g \odot \beta_{GxC} %2B c %2B u %2B \epsilon">,

where 

<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC} \sim \mathcal{N} (0, \sigma^2_{GxC}CC^T)">,

<img src="https://render.githubusercontent.com/render/math?math=c \sim \mathcal{N} (0, \sigma^2_{C}CC^T)">,

<img src="https://render.githubusercontent.com/render/math?math=u \sim \mathcal{N} (0, \sigma^2_{KC}(CC^T \odot K))"> and

<img src="https://render.githubusercontent.com/render/math?math=\epsilon \sim \mathcal{N} (0, \sigma^2_n I)">.


## Brief description of the model terms

<!-- In the [usage page](https://limix.github.io/CellRegMap/usage.html) the input files are listed, here we provide a brief description of their significance.  -->
The following terms should be provided as input files:

* **Phenotype vector (<img src="https://render.githubusercontent.com/render/math?math=y">)** - in the linear mixed model, this is the outcome variable. 
In eQTL mapping, this represents expression level of a given gene of interest, across samples. 
The main application of CellRegMap is using scRNA-seq data, in which case this will be a column vector, with length corresponding to the number of cells considered. 
For optimal fit with the model (which assumes a Gaussian distribution) we recommend [quantile normalising](https://github.com/limix/limix/blob/master/limix/qc/_quant_gauss.py) this vector, or at least [standardising](https://github.com/limix/limix/blob/master/limix/qc/_mean_std.py) it.

* **Genotype vector (<img src="https://render.githubusercontent.com/render/math?math=g">)** - SNP vector. 
This represents the genotype of each sample at the genomic locus of interest, and is typically modelled as 0, 1 or 2, representing the number of minor alleles (however, the model can also handle a continuous vector of dosages). 
Note that a genotype file is well defined at the level of donors, and needs to be appropriately [expanded](https://github.com/annacuomo/CellRegMap_analyses/blob/main/endodiff/preprocessing/Expand_genotypes_kinship.ipynb) across cells.
It is also possible to input a matrix G whose columns represent multiple SNPs (<img src="https://render.githubusercontent.com/render/math?math=g">'s) to be tested for that gene (see "Notes" below).

* **Cellular context matrix (<img src="https://render.githubusercontent.com/render/math?math=C">)** - cellular environment/context matrix. 
Rows are cells, columns are values across the different cellular contexts. 
Columns of C can for example be principal components, or other latent factor representations of the data (e.g., using MOFA [1], ZINB-WaVE [2] or LDVAE [3]), binary vector encoding assignment to different cellular groups such as cell types, or any other factor, including environmental exposures, or disease state. 
Best practice is to column-standardise this matrix.

* **Kinship matrix (<img src="https://render.githubusercontent.com/render/math?math=K">)**, or its decomposition (<img src="https://render.githubusercontent.com/render/math?math=hK">, such that <img src="https://render.githubusercontent.com/render/math?math=K = hK @ hK^T">), a sample covariance, often the so-called [kinship](https://www.cog-genomics.org/plink/1.9/distance) (or genetic relationship matrix; GRM) matrix, appropriately [expanded](https://github.com/annacuomo/CellRegMap_analyses/blob/main/endodiff/preprocessing/Expand_genotypes_kinship.ipynb) across cells.
<!-- This can be.. -->

<!-- * **Background matrices (<img src="https://render.githubusercontent.com/render/math?math=L_i">'s)** - decomposition of the covariance matrix from the background term accounting for repeat samples. It can be shown that the covariance matrix <img src="https://render.githubusercontent.com/render/math?math=(CC^T \odot GG^T)"> can be reformulated as <img src="https://render.githubusercontent.com/render/math?math=\sum_i L_i @ L_i^T">, where <img src="https://render.githubusercontent.com/render/math?math=L_i = diag(\sqrt(\lambda_i) v_i) G">, with <img src="https://render.githubusercontent.com/render/math?math=\lambda_i, v_i"> being the eigenvalues and eigenvectors of <img src="https://render.githubusercontent.com/render/math?math=CC^T">. This decomposition allows us to never having to compute the full covariance matrices which can be extremely large, and work on their decomposed form only. A function that allows to directly compute the <img src="https://render.githubusercontent.com/render/math?math=L_i">'s values from <img src="https://render.githubusercontent.com/render/math?math=C"> and <img src="https://render.githubusercontent.com/render/math?math=G"> will be added soon. -->

* **Covariate matrix (<img src="https://render.githubusercontent.com/render/math?math=W">)** - any additional fixed effect terms to include in the model, such as sex or age. 
If no such terms are needed an intercept of ones should be provided.

<!-- * An additional optional input can be a **filter file** containing known eQTLs (i.e., gene-SNP pairs identified as statistical associations) or individual variants (e.g., GWAS hits) to be investigated. If such a set is not available, it is possible to map eQTL from scratch within the pipeline, see the association test described in the [usage page](https://limix.github.io/CellRegMap/usage.html). -->

The following terms will be estimated by the model:

* **SNP effect sizes**, both due to persistent effects (<img src="https://render.githubusercontent.com/render/math?math=\beta_G">) and to GxC interactions (<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC}">) can be estimated using the estimate_betas() function, see [usage page](https://limix.github.io/CellRegMap/usage.html).

* **other inferred parameters** (<img src="https://render.githubusercontent.com/render/math?math=\alpha, \sigma^2"> values) are estimated by the model but not returned as values.

# Notes

## Necessary inputs
The model will not run if one of **y, W, g** or **C** is not provided as input.

* The following terms are absolutely necessary: expression phenotypes (**y**), genotypes (**g**), and cellular contexts (**C**).

* A kinship matrix (**K**; or its decomposition **hK**, such that K = hK @ hK.T) is highly recommended, to appropriately account for sample structure, especially the repeatedness across cells from the same individual.
  * if you do not have access to a GRM, consider providing a block diagonal sample covariance, with blocks corresponding to individuals.
  * If K (or hK) is not provided, CellRegMap becomes equivalent to [StructLMM](https://limix.github.io/CellRegMap/structlmm.html).


* If no covariates (W) are necessary, simply provide a vector of [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) as an intercept term.


## Each SNP-gene pair should be tested independently
The test is run independently for each gene-SNP pair, thus in the model above, **y** and **g** are one-dimensional vectors, representing i) the expression of a single gene and ii) the genotypes at a single SNP, respectively.

* The implementation does allow for multiple SNPs to be tested for a given gene, this can be achieved by providing a matrix G of which each column is a different SNP G=[g_1, .. g_n].
In this case, the model **simply loops over each SNP and tests one at the time**, then returns a list of p-values, one per SNP.

* On the other hand, each gene needs to be tested separately, as CellRegMap cannot take the full expression matrix as input.

As tests are independent, we recommend parallelising as much as possible, for example submitting independent jobs for each chromosome, gene, or even gene-SNP pair.

## Covariates, cell contexts and repeatedness are fixed
W, C, hK (and thus K) remain the same across all tests (_i.e._, across all SNP-gene pairs).

# Dimensionality

Specified dimensionality for each of the terms, where n is the total number of cells:

* **y**: n x 1 (only one gene tested at a time)
* **W**: n x c, where c is the number of fixed effect covariates (_e.g._, age, sex..)
* **C**: n x k, where k is the number of contexts to test for interactions
* **G**: n x s, where s is the number of SNPs to be tested for a given gene
* **hK**: n x p, where p is the number of individuals, decomposition of the n x n kinship matrix K
<!-- * **K**: n x n, or in alternative -->

<!-- All vectors and matrices should be provided as numpy arrays, and there should be no flat arrays. 
If the shape of a vector is (n,) please reshape to (n,1). -->

# Normalization

For optimal model fit, we recommend standardizing or quantile normalizing (to a standard normal distribution) the phenotype vector **y** and column-standardizing the cellular contexts **C**.
Standardization refers to a transformation of a vector to have 0 mean and standard deviation 1. You can use [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) for this task.
Quantile normalization is a rank-normalization which enforces a standard normal distribution of the vector provided.
For an implementation of quantile-normalization see [here](https://github.com/limix/limix/blob/master/limix/qc/_quant_gauss.py).

<!-- # Genotype format -->

# Pseudocells

This approach refers to the action of grouping together small numbers of similar cells into "pseudocells" to reduce issues due to sparsity and speed up computations by reducing sample size.
Existing implementations include [Metacell](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1812-2) and the [micro pooling approach](https://yoseflab.github.io/VISION/articles/micropooling.html) within the [Vision](https://www.nature.com/articles/s41467-019-12235-0) pipeline.
Those approaches do not directly take into account the presence of several genetically distinct donors, which is important here.
To address this, we recommend using one of these approaches for each donor separately.
For an implementation of how we computed meta-cells in the CellRegMap manuscript (for the neuronal differentiation data analysis), see [here](https://github.com/annacuomo/CellRegMap_analyses/blob/main/neuroseq/preprocessing/create_metacells.py).


# Multiple testing correction

Since thousands of tests are typically run, [multiple testing correction](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) of the test p-values is necessary.
Below, we provide guidelines for how to correct for multiple testing for the two main tests implemented in CellRegMap.
Also refer to workflow [here](https://github.com/annacuomo/CellRegMap_analyses/blob/main/endodiff/usage/README.md)

## Association test

Run discovery, two-step multiple testing correction, 1) within gene across SNPs (FWER), 2) across genes (FDR).
<!-- Mention lenient threshold prior to interaction test -->

## Interaction test

Only one SNP per gene, or at least independent. 
If one SNP per gene straight to step 2 (FDR), if multiple but independent Bonferroni as step 1, then step 2.

# References

[1] Argelaguet\*, Velten\* et al., Molecular Systems Biology, 2018 (MOFA: multi-omics factor analysis) - [link](https://www.embopress.org/doi/full/10.15252/msb.20178124)

[2] Risso et al, Nature Communications, 2018 (ZINB-WaVE: zero-inflated negative binomial-based Wanted Variation Extraction) - [link](https://www.nature.com/articles/s41467-017-02554-5)

[3] Svensson et al, Bioinformatics, 2020 (LDVAE: linearly decoded variational autoencoder) - [link](https://academic.oup.com/bioinformatics/article/36/11/3418/5807606)




 

