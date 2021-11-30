---
layout: default
title: "Usage"
---

There are three main functions that can be run within the CellRegMap package:

* association test
* interaction test
* estimation of effect sizes

## Association test
The main functionality of CellRegMap is to investigate GxC interactions and identify context-specific effects (see Interaction test below). However, to improve scalability, we recommend running the main (and computationally more intensive) function only on a set of candidate eQTLs. In the main paper we consider eQTL previously identified in the original study, however it is also possible to test for persistent eQTL effects within the CellRegMap framework itself, using this function.

The model is the same except background term.

function: scan_association()

## Interaction test

function: scan_interaction()

## Estimation of effect sizes

function: predict_interaction()

The model is the same except background term.

For more details on the functions above I will refer the reader to the Supplementary Methods available with the paper.

## Running the model (simple simulated data)

    import numpy as np
    from numpy.random import RandomState
    from numpy_sugar import ddot
    from numpy_sugar.linalg import economic_svd
    
    from cellregmap import CellRegMap
    
    random = RandomState(1)
    n = 30                               # number of samples (cells)
    p = 5                                # number of individuals
    k = 4                                # number of contexts
    y = random.randn(n, 1)               # outcome vector (expression phenotype)
    E = random.randn(n, k)               # context matrix  
    M = ones((n, 1))                     # intercept (covariate matrix)
    hK = random.randn(n, p)              # decomposition of kinship matrix (K= hK @ hK.T)
    g = 1.0 * (random.rand(n, 1) < 0.2)  # SNP vector
    
    M = concatenate([M, g], axis=1)
    # get eigendecomposition of EEt
    [U, S, _] = economic_svd(E)
    us = U * S
    # get decomposition of K*EEt
    Ls = [ddot(us[:,i], hK) for i in range(us.shape[1])]
    
    # fit null model
    slmm2 = StructLMM2(y, M, E, Ls)
    # Interaction test
    pv = lmm.scan_interaction(g)
    
    print(pv)
    0.6781100453132024


## Downstream analysis (simple simulated data)

## Interpreting the results

## Required dependencies

 

