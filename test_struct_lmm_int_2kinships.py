#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from struct-lmm2 import struct_lmm_int_2kinships
# from qtl_utilities import merge_QTL_results
# import subprocess
# import numpy as np
# import pandas as pd
# import pytest
# import h5py



def test_QTL_analysis():
    '''Run a set of test cases'''
    data_path = '../geuvadis_CEU_test_data/'
    covariates_filename = data_path+'Expression/Geuvadis_CEU_YRI_covariates.txt'
    env_filename = data_path+'Expression/Geuvadis_CEU_YRI_covariates.txt'
    geno_prefix = data_path+'Genotypes/Geuvadis'
    pheno_filename = data_path+'Expression/Geuvadis_CEU_YRI_Expr.txt.gz'
    anno_filename = data_path+'Expression/Geuvadis_CEU_Annot_small.txt'
    kinship_filename= data_path+'Genotypes/Geuvadis_chr1_kinship.normalized.txt'
    individual2sample_filename = data_path + 'Geuvadis_CEU_gte.txt'
    min_maf = 0.25
    min_hwe_P=0.01
    min_call_rate =0.95
    blocksize = 50
    #output_dir = data_path+'limix_QTL_results_kinship_covs_struct/'
    output_dir = '/Users/acuomo/Documents/PhD/test_results/'
    randomSeed = 73
    chromosome = '1'
    ws = 25000 
    run_structLMM_QTL_analysis(pheno_filename = pheno_filename, anno_filename = anno_filename, 
                    env_filename = env_filename, geno_prefix = geno_prefix, plinkGenotype = True, 
                    output_dir = output_dir, window_size = ws, min_maf = min_maf, 
                    min_hwe_P = min_hwe_P, min_call_rate = min_call_rate, association_mode = True,
                    cis_mode = True, blocksize = blocksize, 
                    seed = randomSeed,  n_perm = 10, write_permutations = True, 
                    genetic_range = chromosome, covariates_filename = covariates_filename, kinship_filename = kinship_filename, 
                    sample_mapping_filename = individual2sample_filename)


if __name__=='__main__':
    test_QTL_analysis()
