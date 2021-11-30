---
layout: default
title: "Input Files"
mathjax: true
---

## Brief description

In the [usage page](https://limix.github.io/CellRegMap/usage.html) the input files are listed, here we provide a brief description of their significance. 

The CellRegMap can be cast as:

<img src="https://render.githubusercontent.com/render/math?math=y = W\alpha %2B g\beta_G %2B g \odot \beta_{GxC} %2B c %2B u %2B \epsilon">

where 
<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC} \sim \mathcal{N} (0, \sigma^2_{GxC}CC^T)">
<img src="https://render.githubusercontent.com/render/math?math=c \sim \mathcal{N} (0, \sigma^2_{C}CC^T)">
<img src="https://render.githubusercontent.com/render/math?math=u \sim \mathcal{N} (0, \sigma^2_{KC}(CC^T@GG^T))">
<img src="https://render.githubusercontent.com/render/math?math=\epsilon \sim \mathcal{N} (0, \sigma^2_n I)">

* **Phenotype file (y)** - in the linear mixed model, this is the outcome variable. In eQTL mapping, this represents expression level of a given gene of interest, across samples. The main application of CellRegMap is using scRNA-seq data, in which case this will be a column vector, with length corresponding to the number of cells considered.
* **Genotype file (g)** - SNP vector. This represents the genotype of each sample at the genomic locus of interest, and is typically modelled as 0, 1 or 2, representing the number of minor alleles (however it can also be a continuous vector of dosages). Note that a genotype file is well defined at the level of donors, and needs to be appropriately expanded across cells.
* **Cellular context file (E)** - cellular environments/context matrix. Rows are cells, colunms are values across the different cellular contexts. Columns of E can for example be principal components, or other latent factor representations of the data, or in alternative binary vector encoding assignment to different cellular groups such as cell types.
* **Covariate file (M)** -

## Preparing input files (general guidelines)



 

