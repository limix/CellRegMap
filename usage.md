---
layout: default
title: "Usage"
---

There are three main functions that can be run within the CellRegMap package:

* Association test 
* Interaction test
* Estimation of effect sizes

## Association test (persistent effects)
The main functionality of CellRegMap is to investigate genotype-context (GxC) interactions and identify context-specific genetic effects on expression in cohort-scale single-cell data (see **Interaction test** below). 
However, to improve scalability, we recommend running the (computationally more intensive) interaction-test function only on a set of candidate eQTLs. 
In the [original CellRegMap paper](https://www.biorxiv.org/content/10.1101/2021.09.01.458524v1) we consider eQTLs previously identified in the original studies, but it is now also possible to test for persistent eQTL effects within the CellRegMap framework itself, using the association-test function. 
In this case, the model can be cast as:

<img src="https://render.githubusercontent.com/render/math?math=y = W\alpha %2B g\beta_G %2B c %2B u %2B \epsilon">,

which is similar to the main model except for the GxC term, which is missing. Here, we test for a persistent effect only, i.e., <img src="https://render.githubusercontent.com/render/math?math=\beta_G \neq 0">.

CellRegMap function: _run_association()_

## Interaction test (GxC effects)
This is the main test implemented in CellRegMap, where we test for GxC effects across cellular states and individual SNP variants. 
In this case we consider the full model:

<img src="https://render.githubusercontent.com/render/math?math=y = W\alpha %2B g\beta_G %2B g \odot \beta_{GxC} %2B c %2B u %2B \epsilon"> 

and test for <img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC} \neq 0">.
While in principal any SNP-gene pairs can be tested for GxC effects, we recommend running this test on a set of candidate eQTLs (either known a priori or identified using the **Association test** described above), or interesting (e.g., disease-linked) variants to improve statistical power.

CellRegMap function: _run_interaction()_

## Estimation of effect sizes
Finally, CellRegMap can be used to estimate cell-level effect sizes driven by GxC effects for individual eQTLs (<img src="https://render.githubusercontent.com/render/math?math=\beta_{GxC}">), thus predicting the cells where those effects are detected. 
Generally, it makes sense to use this function to characterise eQTLs that show evidence of GxC effects, i.e., for which significant p-values were obtained when using the **Interaction test** above. 
The model is the same except for the term <img src="https://render.githubusercontent.com/render/math?math=c">, which is now modelled as fixed effects in order to estimate the GxC term itself.

CellRegMap function: _estimate_betas()_

For more details on the tests above and underlying assumptions we refer the reader to the Supplementary Methods available as part of the [paper's Supplementary Material](https://www.biorxiv.org/content/10.1101/2021.09.01.458524v1.supplementary-material).

## Simple usage example

The model is implemented in [python](https://www.python.org).
All vectors and matrices should be provided as [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html), and there should be no flat (one-dimensional) arrays. 
If the shape of a vector is (n,) please [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) to (n,1).

Below, see a simple usage example with toy inputs:

    from numpy import ones
    from numpy.random import RandomState
    
    from cellregmap import run_association, run_interaction, estimate_betas
    
    random = RandomState(1)
    n = 30                               # number of samples (cells)
    p = 5                                # number of individuals
    k = 4                                # number of contexts
    y = random.randn(n, 1)               # outcome vector (expression phenotype, one gene only)
    C = random.randn(n, k)               # context matrix  
    W = ones((n, 1))                     # intercept (covariate matrix)
    hK = random.randn(n, p)              # decomposition of kinship matrix (K = hK @ hK.T)
    g = 1.0 * (random.rand(n, 1) < 0.2)  # SNP vector
    
    ## Association test
    pv0 = run_association(y, W, C, g, hK=hK)[0]
    print(f'Association test p-value: {pv0}')
    
    ## Interaction test
    pv = run_interaction(y, W, C, g, hK=hK)[0]
    print(f'Interaction test p-value: {pv}')
    
    # Effect sizes estimation
    betas = estimate_betas(y, W, C, g, hK=hK)
    beta_G = betas[0]                         # persistent effect (scalar)
    beta_GxC = betas[1][0]                    # GxC effects (vector)
    print(f'persistent genetic effect (betaG): {betaG}')
    print(f'cell-level effect sizes due to GxC (betaGxC): {betaGxC}')

<!-- ## Downstream analysis (simple simulated data)

## Interpreting the results

## Required dependencies -->

 

