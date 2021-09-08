---
layout: default
title: "StructLMM"
---

## Relation to StructLMM model
CellRegMap builds and extends the structured linear mixed model (StructLMM) model, proposed in [Moore et al, 2018](https://www.nature.com/articles/s41588-018-0271-0), in the context of population genetics. StructLMM allows to...

However, StructLMM has one limitation.. CellRegMap overcomes this by 

The original StructLMM model can be run using CellRegMap, by simply setting the relatedness matrix to None, i.e.: 

    K=None
    
and then.. 

    s = CellRegMap()
   
However, if there is any relatedness, population stratification etc we recommend to use the updated model.
