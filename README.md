# CellRegMap

Cellular Regulatory Map, a linear mixed model approach to perform multi-context eQTL mapping leveraging single cell RNA sequencing (scRNA-seq) data.
Similar to [StructLMM](https://www.nature.com/articles/s41588-018-0271-0) but importantly now accounting for sample structure, including population structure and repeated observations for the same samples, e.g., multiple cells for the same donor.

The CellRegMap model and its applications to both real and simulated data are described in the CellRegMap manuscript.  

## Install

From your command line, enter

    pip install cellregmap

in your command line.

## Development

To install it in development mode, enter

    git clone https://github.com/limix/CellRegMap.git
    cd CellRegMap
    pip install -e .

in your command line.

## Running tests

From your command line, enter

    python setup.py test

## Project layout

    ├─ old_files/       old scripts
    ├─ references/      documents on the mathematical concepts
    └─ CellRegMap/      package implementation
       └─ test/         test file

## References

- [Exploring Multivariate Gene-Environment Interactions: Models And Applications](https://www.repository.cam.ac.uk/handle/1810/290971)
- [Optimal tests for rare variant effects in sequencing association studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3440237/) [Supplementary material](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3440237/bin/supp_kxs014_kxs014supp.pdf)
