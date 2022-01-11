.. CellRegMap documentation master file, created by
   sphinx-quickstart on Mon Jan 10 23:46:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CellRegMap's documentation!
======================================

What is CellRegMap
------------------

The cellular regulatory map (`cellregmap``) is a statistical framework to map context-specific effects of genetic variants on single-cell gene expression.
It builds on a linear mixed model and is implemented in Python.

What do you need to run CellRegMap
----------------------------------

Single-cell expression profiles assayed through scRNA-seq in the form of a count matrix (cells x genes)
Cellular contexts: these can be known factors or a latent representation of the space (e.g., PCs by cells)
Genotypes: CellRegMap tests for individual eQTL effects between a genetic variant and a gene's expression, so genotypes need to be provided for variants of interest.
A kinship matrix accounting for relatedness among samples. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   functions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
