---
layout: default
title: "Input Files"
mathjax: true
---

# The CellRegMap model

The CellRegMap can be cast as:

<img src="https://render.githubusercontent.com/render/math?math=y = W\alpha %2B g\beta_G %2B g \odot \beta_{GxC} %2B c %2B u %2B \epsilon">,

where 

<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC} \sim \mathcal{N} (0, \sigma^2_{GxC}CC^T)">,

<img src="https://render.githubusercontent.com/render/math?math=c \sim \mathcal{N} (0, \sigma^2_{C}CC^T)">,

<img src="https://render.githubusercontent.com/render/math?math=u \sim \mathcal{N} (0, \sigma^2_{KC}(CC^T \odot K))"> and

<img src="https://render.githubusercontent.com/render/math?math=\epsilon \sim \mathcal{N} (0, \sigma^2_n I)">.

# Brief description of the model terms

In the [usage page](https://limix.github.io/CellRegMap/usage.html) the input files are listed, here we provide a brief description of their significance. 
The following terms should be provided as input files, where n is the total number of cells:

* **Phenotype vector (<img src="https://render.githubusercontent.com/render/math?math=y">)** - in the linear mixed model, this is the outcome variable. In eQTL mapping, this represents expression level of a given gene of interest, across samples. The main application of CellRegMap is using scRNA-seq data, in which case this will be a column vector, with length corresponding to the number of cells considered. For optimal fit with the model (which assumes a Gaussian distribution) we recommend [quantile normalising](https://github.com/limix/limix/blob/master/limix/qc/_quant_gauss.py) this vector, or at least standardising it.

* **Genotype vector (<img src="https://render.githubusercontent.com/render/math?math=g">)** - SNP vector. This represents the genotype of each sample at the genomic locus of interest, and is typically modelled as 0, 1 or 2, representing the number of minor alleles (however the model can also handle a continuous vector of dosages). Note that a genotype file is well defined at the level of donors, and needs to be appropriately expanded across cells.

* **Cellular context matrix (<img src="https://render.githubusercontent.com/render/math?math=C">)** - cellular environment/context matrix. Rows are cells, columns are values across the different cellular contexts. Columns of E can for example be principal components, or other latent factor representations of the data, or in alternative binary vector encoding assignment to different cellular groups such as cell types. Best practice is to column-standardise this matrix.

* **Kinship matrix (<img src="https://render.githubusercontent.com/render/math?math=K">)**, or its decomposition (<img src="https://render.githubusercontent.com/render/math?math=hK">, such that <img src="https://render.githubusercontent.com/render/math?math=K = hK @ hK^T">), a kinship or relatedness matrix, modelling the similarity across individuals, then expanded across cells.

<!-- * **Background matrices (<img src="https://render.githubusercontent.com/render/math?math=L_i">'s)** - decomposition of the covariance matrix from the background term accounting for repeat samples. It can be shown that the covariance matrix <img src="https://render.githubusercontent.com/render/math?math=(CC^T \odot GG^T)"> can be reformulated as <img src="https://render.githubusercontent.com/render/math?math=\sum_i L_i @ L_i^T">, where <img src="https://render.githubusercontent.com/render/math?math=L_i = diag(\sqrt(\lambda_i) v_i) G">, with <img src="https://render.githubusercontent.com/render/math?math=\lambda_i, v_i"> being the eigenvalues and eigenvectors of <img src="https://render.githubusercontent.com/render/math?math=CC^T">. This decomposition allows us to never having to compute the full covariance matrices which can be extremely large, and work on their decomposed form only. A function that allows to directly compute the <img src="https://render.githubusercontent.com/render/math?math=L_i">'s values from <img src="https://render.githubusercontent.com/render/math?math=C"> and <img src="https://render.githubusercontent.com/render/math?math=G"> will be added soon. -->

* **Covariate matrix (<img src="https://render.githubusercontent.com/render/math?math=W">)** - any additional fixed effect terms to include in the model, such as sex or age. If no such terms are needed an intercept of ones should be provided.

* An additional optional input can be a **filter file** containing known eQTLs (i.e., gene-SNP pairs identified as statistical associations) or individual variants (e.g., GWAS hits) to be investigated. If such a set is not available, it is possible to map eQTL from scratch within the pipeline, see the association test described in the [usage page](https://limix.github.io/CellRegMap/usage.html).

The following terms will be estimated by the model:

* **SNP effect sizes**, both due to persistent effects (<img src="https://render.githubusercontent.com/render/math?math=\beta_G">) and to GxC interactions (<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC}">) can be estimated using the estimate_betas() function, see [usage page](https://limix.github.io/CellRegMap/usage.html).

* **other inferred parameters** (<img src="https://render.githubusercontent.com/render/math?math=\alpha, \sigma^2"> values) are estimated by the model but not returned as values.

# Dimensionality

Specified dimensionality for each of the terms, where n is the total number of cells:

* **y**: n x 1 (only one gene tested at a time)
* **W**: n x c, where c is the number of fixed effect covariates (e.g., age, sex..)
* **C**: n x k, where k is the number of contexts to test for interactions
* **G**: n x s, where s is the number of SNPs to be tested for a given gene
* **K**: n x n, or in alternative
* **hK**, its decomposition: n x p, where p is the number of individuals

All vectors and matrices should be provided as numpy arrays, and there should be no flat arrays. 
If the shape of a vector is (n,) please reshape to (n,1).

# Normalization

For optimal model fit, we recommend standardizing or quantile normalizing (to a standard normal distribution) the phenotype vector **y** and column-standardizing the cellular contexts **C**.
Standardization refers to a transformation of a vector to have 0 mean and standard deviation 1. You can use [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) for this task.
Quantile normalization is a rank-normalization which enforces a standard normal distribution of the vector provided.
For an implementation of quantile-normalization see [here](https://github.com/limix/limix/blob/master/limix/qc/_quant_gauss.py).

# Pseudocells

This approach refers to grouping together small numbers of similar cells into "pseudocells" to reduce issues due to sparsity and speed computations by reducing sample size.
Exisiting implementations include [Metacell](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1812-2) and the [micro pooling approach](https://yoseflab.github.io/VISION/articles/micropooling.html) within the [Vision](https://www.nature.com/articles/s41467-019-12235-0) pipeline.

<!-- If many cells + sparse, pseudocells / metacells may be preferable - add references. -->

# Multiple testing correction

## Association test

Run discovery, two-step multiple testing correction, 1) within gene across SNPs, 2) across genes.
Mention lenient threshold prior to interaction test

## Interaction test

Only one SNP per gene, or at least independent. If one SNP per gene straight to step 2, if multiple but independent Bonferroni as step 1, then step 2.

<!-- ## Preparing input files (general guidelines) -->



 

