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

<img src="https://render.githubusercontent.com/render/math?math=u \sim \mathcal{N} (0, \sigma^2_{KC}(CC^T@GG^T))"> and

<img src="https://render.githubusercontent.com/render/math?math=\epsilon \sim \mathcal{N} (0, \sigma^2_n I)">.

The following terms should be provided as input files:

* **Phenotype vector (<img src="https://render.githubusercontent.com/render/math?math=y">)** - in the linear mixed model, this is the outcome variable. In eQTL mapping, this represents expression level of a given gene of interest, across samples. The main application of CellRegMap is using scRNA-seq data, in which case this will be a column vector, with length corresponding to the number of cells considered.

* **Genotype vector (<img src="https://render.githubusercontent.com/render/math?math=g">)** - SNP vector. This represents the genotype of each sample at the genomic locus of interest, and is typically modelled as 0, 1 or 2, representing the number of minor alleles (however it can also be a continuous vector of dosages). Note that a genotype file is well defined at the level of donors, and needs to be appropriately expanded across cells.

* **Cellular context matrix (<img src="https://render.githubusercontent.com/render/math?math=C">)** - cellular environments/context matrix. Rows are cells, colunms are values across the different cellular contexts. Columns of E can for example be principal components, or other latent factor representations of the data, or in alternative binary vector encoding assignment to different cellular groups such as cell types.

* **Covariate matrix (<img src="https://render.githubusercontent.com/render/math?math=W">)** - any additional fixed effect terms to include in the model, such as sex or age. If not such terms are needed an intercept of ones should be provided.

The following terms will be estimated by the model:

* SNP effect size, both due to persistent effects (<img src="https://render.githubusercontent.com/render/math?math=\beta_G">) and to GxC interactions (<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC}">) can be estimated using the predict_interaction() function

* other paramters (\alpha, \sigma^2 values) are estimated by the model but not returned as values.

<!-- ## Preparing input files (general guidelines) -->



 

