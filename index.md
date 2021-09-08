---
layout: default
title: CellRegMap
---

# CellRegMap

Cellular Regulatory Map (CellRegMap) is a linear mixed model approach to test for and characterize context-specific eQTL variants across several cell contexts and states.
CellRegMap leverages single cell RNA sequencing (scRNA-seq) data, and does not require discretization of cells into cell groups.
CellRegMap builds on the previously proposed [StructLMM](https://www.nature.com/articles/s41588-018-0271-0) but importantly now accounting for sample structure, including population structure and repeated observations for the same samples, e.g., multiple cells for the same donor.

![Fig1](https://user-images.githubusercontent.com/25035866/132485517-0e111014-1b80-42f1-bd8b-b9ca0750bd23.png)

For more details on the CellRegMap model and its applications to both real and simulated data, see our [preprint](https://www.biorxiv.org/content/10.1101/2021.09.01.458524v1).  
