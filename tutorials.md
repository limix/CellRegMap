---
layout: default
title: "Tutorials"
---

# Usage

## Running the model (simple simulated data)

    import numpy as np
    from numpy.random import RandomState
    
    from cellregmap import CellRegMap
    
    random = RandomState(1)
    n = 20
    k = 4
    y = random.randn(n, 1)
    E = random.randn(n, k)
    M = ones((n, 1))
    g = 1.0 * (random.rand(n, 1) < 0.2)
    
    M = concatenate([M, g], axis=1)
    lmm = CellRegMap(y, M, E)
    lmm.fit(verbose=False)
    # Interaction test
    pv = lmm.scan_interaction(g)
    
    print(pv)
    0.6781100453132024


## Downstream analysis (simple simulated data)

## Interpreting the results

## Preparing input files (general guidelines)

* Phenotype file
* Genotype file
* Cellular context file
* Additional files

 
