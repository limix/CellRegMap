---
layout: default
title: "StructLMM"
---

## Relation to StructLMM model
CellRegMap builds on and extends the structured linear mixed model (StructLMM) model, proposed in [Moore et al, 2018](https://www.nature.com/articles/s41588-018-0271-0), in the context of population genetics. StructLMM allows to test for GxE effects across multiple environmental exposures at once, extending traditional interaction models which can only consider one environment at a time.

However, StructLMM is not designed to deal with repeated or related samples. Thus, it is not well suited to model longitudinal data (where multiple observations from the same individuals are collected over time) or single-cell data (where multiple cells are collected from the same individual), nor can it optimally model population stratification and cryptic relatedness, which have been shown to be prevalent in population genetic data.
CellRegMap overcomes this by including an additional random effect term that models relatedness across samples.

Nevertheless, the original StructLMM model can be run using CellRegMap, by simply setting the relatedness matrix to None, i.e.: 

    K=None
    
and then following the instructions from the [Usage](https://limix.github.io/CellRegMap/usage.html) or [Tutorials](https://limix.github.io/CellRegMap/tutorials.html) pages.
