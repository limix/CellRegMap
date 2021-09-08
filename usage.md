---
layout: default
title: "Usage"
---

## Running the model (simple simulated data)

    import numpy as np
    from numpy.random import RandomState
    
    from cellregmap import CellRegMap
    
    random = RandomState(1)
    n = 30                               # number of samples (cells)
    p = 5                                # number of individuals
    k = 4                                # number of contexts
    y = random.randn(n, 1)               # outcome vector (expression phenotype)
    E = random.randn(n, k)               # context matrix  
    M = ones((n, 1))                     # intercept (covariate matrix)
    Ls = random.randn(n, k)              # decomposition of kinship matrix (improve)
    g = 1.0 * (random.rand(n, 1) < 0.2)  # SNP vector
    
    M = concatenate([M, g], axis=1)
    
    # fit null model
    slmm2 = StructLMM2(y, M, E, Ls)
    # Interaction test
    pv = lmm.scan_interaction(g)
    
    print(pv)
    0.6781100453132024


## Downstream analysis (simple simulated data)

## Interpreting the results

## Required dependencies

 

