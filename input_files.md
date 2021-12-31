---
layout: default
title: "Input Files"
mathjax: true
---

## Brief description

In the [usage page](https://limix.github.io/CellRegMap/usage.html) the input files are listed, here we provide a brief description of their significance. 

The CellRegMap can be cast as:

<img src="https://render.githubusercontent.com/render/math?math=y = W\alpha %2B g\beta_G %2B g \odot \beta_{GxC} %2B c %2B u %2B \epsilon">,

where 

<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC} \sim \mathcal{N} (0, \sigma^2_{GxC}CC^T)">,

<img src="https://render.githubusercontent.com/render/math?math=c \sim \mathcal{N} (0, \sigma^2_{C}CC^T)">,

<img src="https://render.githubusercontent.com/render/math?math=u \sim \mathcal{N} (0, \sigma^2_{KC}(CC^T \odot GG^T))"> and

<img src="https://render.githubusercontent.com/render/math?math=\epsilon \sim \mathcal{N} (0, \sigma^2_n I)">.

The following terms should be provided as input files, where n is the total number of cells:

* **Phenotype vector (<img src="https://render.githubusercontent.com/render/math?math=y">)** - in the linear mixed model, this is the outcome variable. In eQTL mapping, this represents expression level of a given gene of interest, across samples. The main application of CellRegMap is using scRNA-seq data, in which case this will be a column vector, with length corresponding to the number of cells considered. For optimal fit with the model (which assumes a Gaussian distribution) we recommend [quantile normalising](https://github.com/limix/limix/blob/master/limix/qc/_quant_gauss.py) this vector, or at least standardising it.

dimensionality: this is an **n x 1** vector, where n is the number of cells. A different model needs to be fitted for each gene. 

* **Genotype vector (<img src="https://render.githubusercontent.com/render/math?math=g">)** - SNP vector. This represents the genotype of each sample at the genomic locus of interest, and is typically modelled as 0, 1 or 2, representing the number of minor alleles (however the model can also handle a continuous vector of dosages). Note that a genotype file is well defined at the level of donors, and needs to be appropriately expanded across cells.

* **Cellular context matrix (<img src="https://render.githubusercontent.com/render/math?math=C">)** - cellular environment/context matrix. Rows are cells, colunms are values across the different cellular contexts. Columns of E can for example be principal components, or other latent factor representations of the data, or in alternative binary vector encoding assignment to different cellular groups such as cell types. Best practice is to column-standardise this matrix.

* **Background matrices (<img src="https://render.githubusercontent.com/render/math?math=L_i">'s)** - decomposition of the covariance matrix from the background term accounting for repeat samples. It can be shown that the covariance matrix <img src="https://render.githubusercontent.com/render/math?math=(CC^T \odot GG^T)"> can be reformulated as <img src="https://render.githubusercontent.com/render/math?math=\sum_i L_i @ L_i^T">, where <img src="https://render.githubusercontent.com/render/math?math=L_i = diag(\sqrt(\lambda_i) v_i) G">, with <img src="https://render.githubusercontent.com/render/math?math=\lambda_i, v_i"> being the eigenvalues and eigenvectors of <img src="https://render.githubusercontent.com/render/math?math=CC^T">. This decomposition allows us to never having to compute the full covariance matrices which can be extremely large, and work on their decomposed form only. A function that allows to directly compute the <img src="https://render.githubusercontent.com/render/math?math=L_i">'s values from <img src="https://render.githubusercontent.com/render/math?math=C"> and <img src="https://render.githubusercontent.com/render/math?math=G"> will be added soon.

* **Covariate matrix (<img src="https://render.githubusercontent.com/render/math?math=W">)** - any additional fixed effect terms to include in the model, such as sex or age. If no such terms are needed an intercept of ones should be provided.

* An additional optional input can be a **filter file** containing known eQTLs (i.e., gene-SNP pairs identified as statistical associations) or individual variants (e.g., GWAS hits) to be investigated. If such a set is not available, it is possible to map eQTL from scratch within the pipeline, see the association test described in the [usage page](https://limix.github.io/CellRegMap/usage.html).

The following terms will be estimated by the model:

* **SNP effect sizes**, both due to persistent effects (<img src="https://render.githubusercontent.com/render/math?math=\beta_G">) and to GxC interactions (<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC}">) can be estimated using the predict_interaction() function, see [usage page](https://limix.github.io/CellRegMap/usage.html).

* **other inferred parameters** (<img src="https://render.githubusercontent.com/render/math?math=\alpha, \sigma^2"> values) are estimated by the model but not returned as values.

<!-- ## Preparing input files (general guidelines) -->



 

