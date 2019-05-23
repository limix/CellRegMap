
import sys
sys.path.insert(0,'/hps/nobackup/stegle/users/mjbonder/tools/hipsci_pipeline/limix_QTL_pipeline/')

# import glob
# import os
# # from subprocess import run
# import pandas as pd
# import re
# from os.path import join
# import scipy as sp
import numpy as np
# # from struct_lmm.interpretation import PredictGenEffect
from sklearn.preprocessing import Imputer
import limix
# import qtl_output
import qtl_loader_utils
# import qtl_parse_args
# import qtl_utilities as utils
# from bgen_reader import read_bgen
# from numpy import linalg as LA
# import scipy.linalg as la
from tqdm import tqdm 

from limix_core.covar import FreeFormCov
from limix_core.gp import GP2KronSumLR

def _mean_standardize(X, axis = None):
    from numpy_sugar import epsilon
    from numpy import inf

    X = X - np.nanmean(X,0)
    X = X / np.nanstd(X,0)

    return X

def get_genotype_data(geno_prefix):
    bim,fam,bed = limix.io.plink.read(geno_prefix, verbose = False)
    fam.set_index('iid', inplace = True)
    return bim,fam,bed

def run_structLMM2_int_load_intersect_phenotype_environments_covariates_kinship_sample_mapping\
        (pheno_filename, anno_filename, env_filename, geno_prefix, plinkGenotype,  
            cis_mode = True, interaction_mode = True, 
            relatedness_score = 0.95, snps_filename = None, feature_filename = None, 
            snp_feature_filename = None, selection = 'all', covariates_filename = None, kinship_filename = None, 
            sample_mapping_filename = None, feature_variant_covariate_filename = None):  
    selectionStart = None
    selectionEnd = None
    if(":" in selection):
        parts = selection.split(":")
        if("-" not in parts[1]):
            print("No correct sub selection.")
            print("Given in: "+selection)
            print("Expected format: (chr number):(start location)-(stop location)")
            sys.exit()
        chromosome = parts[0]
        if("-" in parts[1]):
            parts2 = parts[1].split("-") 
            selectionStart = int(parts2[0])
            selectionEnd = int(parts2[1])
    else :
        chromosome = selection

    ''' function to take input and intersect sample and genotype.'''
    #Load input data files & filter for relevant data
    #Load input data filesf

    phenotype_df = qtl_loader_utils.get_phenotype_df(pheno_filename)
    annotation_df = qtl_loader_utils.get_annotation_df(anno_filename)

    if(plinkGenotype):
        bim,fam,bed = get_genotype_data(geno_prefix)
        annotation_df.replace(['X', 'Y', 'XY', 'MT'], ['23', '24', '25', '26'], inplace = True)
        if chromosome == 'X' :
            chromosome = '23'
        elif chromosome == 'Y':
            chromosome = '24'
        elif chromosome == 'XY':
            chromosome = '25'
        elif chromosome == 'MT':
            chromosome = '26'


    else :
        geno_prefix+='.bgen'
        print(geno_prefix)
    print("Intersecting data.")

    if(annotation_df.shape[0] != annotation_df.groupby(annotation_df.index).first().shape[0]): 
        print("Only one location per feature supported. If multiple locations are needed please look at: --extended_anno_file")
        sys.exit()

    ##Make sure that there is only one entry per feature id!.
    sample2individual_df = qtl_loader_utils.get_samplemapping_df(sample_mapping_filename, list(phenotype_df.columns), 'sample')
    sample2individual_df['sample'] = sample2individual_df.index
    sample2individual_df = sample2individual_df.drop_duplicates();


    ##Filter first the linking files!
    #Subset linking to relevant genotypes.
    orgSize = sample2individual_df.shape[0]
    sample2individual_df = sample2individual_df.loc[sample2individual_df['iid'].map(lambda x: x in list(map(str, fam.index))),:]
    diff = orgSize - sample2individual_df.shape[0]
    orgSize = sample2individual_df.shape[0]
    print("Dropped: "+str(diff)+" samples because they are not present in the genotype file.")
    
    #Subset linking to relevant phenotypes.
    sample2individual_df = sample2individual_df.loc[np.intersect1d(sample2individual_df.index,phenotype_df.columns),:]
    diff = orgSize- sample2individual_df.shape[0]
    orgSize = sample2individual_df.shape[0]
    print("Dropped: " + str(diff) + " samples because they are not present in the phenotype file.")
    #Subset linking vs kinship.
    kinship_df = qtl_loader_utils.get_kinship_df(kinship_filename)
    if kinship_df is not None:
        #Filter from individual2sample_df & sample2individual_df since we don't want to filter from the genotypes.
        sample2individual_df = sample2individual_df[sample2individual_df['iid'].map(lambda x: x in list(map(str, kinship_df.index)))]
        diff = orgSize - sample2individual_df.shape[0]
        orgSize = sample2individual_df.shape[0]
        print("Dropped: " + str(diff) + " samples because they are not present in the kinship file.")
    #Subset linking vs covariates.
    covariate_df = qtl_loader_utils.get_covariate_df(covariates_filename)
    if covariate_df is not None:
        if np.nansum(covariate_df == 1, 0).max() < covariate_df.shape[0]: covariate_df.insert(0, 'ones',np.ones(covariate_df.shape[0]))
        sample2individual_df = sample2individual_df.loc[list(set(sample2individual_df.index) & set(covariate_df.index)),:]
        diff = orgSize - sample2individual_df.shape[0]
        orgSize = sample2individual_df.shape[0]
        print("Dropped: " + str(diff) + " samples because they are not present in the covariate file.")
    #Subset linking vs environments.
    environment_df = qtl_loader_utils.get_env_df(env_filename)
    # import pdb; pdb.set_trace()
    # if np.nansum(environment_df == 1, 0).max() < environment_df.shape[0]: environment_df.insert(0, 'ones',np.ones(environment_df.shape[0]))
    sample2individual_df = sample2individual_df.loc[list(set(sample2individual_df.index) & set(environment_df.index)),:]
    diff = orgSize - sample2individual_df.shape[0]
    orgSize = sample2individual_df.shape[0]
    print("Dropped: " + str(diff) + " samples because they are not present in the environment file.")

    ###
    print("Number of samples with genotype & phenotype data: " + str(sample2individual_df.shape[0]))


    ##Filter now the actual data!
    #Filter phenotype data based on the linking files.
    phenotype_df = phenotype_df.loc[list(set(phenotype_df.index) & set(annotation_df.index)), sample2individual_df.index.values]

    #Filter kinship data based on the linking files.
    genetically_unique_individuals = None
    if kinship_df is not None:
        kinship_df = kinship_df.loc[np.intersect1d(kinship_df.index, sample2individual_df['iid']), np.intersect1d(kinship_df.index, sample2individual_df['iid'])]
        genetically_unique_individuals = utils.get_unique_genetic_samples(kinship_df, relatedness_score);

    #Filter covariate data based on the linking files.
    if covariate_df is not None:
        covariate_df = covariate_df.loc[np.intersect1d(covariate_df.index, sample2individual_df.index.values),:]
    
    snp_feature_filter_df = qtl_loader_utils.get_snp_feature_df(snp_feature_filename)
    try:
        feature_filter_df = qtl_loader_utils.get_snp_df(feature_filename)
    except:
        if feature_filename  is not None:
            feature_filter_df = pd.DataFrame(index = feature_filename)
    #Do filtering on features.
    if feature_filter_df is not None:
        phenotype_df = phenotype_df.loc[feature_filter_df.index,:]
        ##Filtering on features to test.
    if snp_feature_filter_df is not None:
        phenotype_df = phenotype_df.loc[np.unique(snp_feature_filter_df['feature']),:]
        ##Filtering on features  to test from the combined feature snp filter.

    if ((not cis_mode) and len(set(bim['chrom'])) < 22) :
        print("Warning, running a trans-analysis on snp data from less than 22 chromosomes.\nTo merge data later the permutation P-values need to be written out.")

    if(cis_mode):
        #Remove features from the annotation that are on chromosomes which are not present anyway.
        annotation_df = annotation_df[np.in1d(annotation_df['chromosome'], list(set(bim['chrom'])))]

    #Prepare to filter on snps.
    snp_filter_df = qtl_loader_utils.get_snp_df(snps_filename)
    if snp_filter_df is not None:
        toSelect = set(snp_filter_df.index).intersection(set(bim['snp']))
        bim = bim.loc[bim['snp'].isin(toSelect)]
        ##Filtering on SNPs to test from the snp filter.

    if snp_feature_filter_df is not None:
        toSelect = set(np.unique(snp_feature_filter_df['snp_id'])).intersection(set(bim['snp']))
        bim = bim.loc[bim['snp'].isin(toSelect)]
        ##Filtering on features  to test from the combined feature snp filter.
    
   
    #Determine features to be tested
    if chromosome=='all':
        feature_list = list(set(annotation_df.index) & set(phenotype_df.index))
    else:
        if not selectionStart is None :
            lowest = min([selectionStart,selectionEnd])
            highest = max([selectionStart,selectionEnd])
            annotation_df['mean'] = ((annotation_df["start"] + annotation_df["end"])/2)
            feature_list = list(set(annotation_df.iloc[(annotation_df['chromosome'].values == chromosome) & (annotation_df['mean'].values >= lowest) & (annotation_df["mean"].values < highest)].index.values) & set(phenotype_df.index))
            del annotation_df['mean']
        else :
            feature_list = list(set(annotation_df[annotation_df['chromosome'] == chromosome].index) & set(phenotype_df.index))

    print("Number of features to be tested: " + str(len(feature_list)))
    print("Total number of variants to be considered, before variante QC and feature intersection: " + str(bim.shape[0]))
    

    feature_variant_covariate_df = qtl_loader_utils.get_snp_feature_df(feature_variant_covariate_filename) 
    return [phenotype_df, kinship_df, covariate_df, environment_df, sample2individual_df, complete_annotation_df, annotation_df, snp_filter_df, snp_feature_filter_df, genetically_unique_individuals, minimum_test_samples, feature_list,bim,fam,bed, chromosome, selectionStart, selectionEnd, feature_variant_covariate_df]


import struct_lmm2 as StructLMM2

def run_structLMM2_int(pheno_filename, anno_filename, env_filename, geno_prefix, output_dir,  plinkGenotype = True, 
                        window_size = 250000, min_maf = 0.05, min_hwe_P = 0.001, min_call_rate = 0.95, 
                        interaction_mode = True, cis_mode = True, gaussianize_method = None,  
                        seed = np.random.randint(40000), relatedness_score = 0.95, 
                        feature_variant_covariate_filename = None, snps_filename = None, feature_filename = None, 
                        snp_feature_filename = None, genetic_range = 'all', covariates_filename = None, 
                        kinship_filename = None, sample_mapping_filename = None):
    fill_NaN = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
        
    [phenotype_df, kinship_df, covariate_df, environment_df, sample2individual_df, annotation_df, snp_filter_df, 
    snp_feature_filter_df, feature_list, bim, fam, bed, 
    chromosome, selectionStart, selectionEnd, feature_variant_covariate_df]=\
    run_structLMM2_int_load_intersect_phenotype_environments_covariates_kinship_sample_mapping(pheno_filename = pheno_filename, 
                anno_filename = anno_filename, env_filename = env_filename, geno_prefix = geno_prefix, 
                plinkGenotype = plinkGenotype, cis_mode = cis_mode, interaction_mode = interaction_mode, 
                relatedness_score = relatedness_score, snps_filename = snps_filename, 
                feature_filename = feature_filename, snp_feature_filename = snp_feature_filename, 
                selection = genetic_range, covariates_filename = covariates_filename, 
                kinship_filename = kinship_filename, sample_mapping_filename = sample_mapping_filename,  
                feature_variant_covariate_filename = feature_variant_covariate_filename)

    #import pdb; pdb.set_trace()
    if(feature_list == None or len(feature_list) == 0):
        print ('No features to be tested.')
        sys.exit()
    
    #Arrays to store number of samples and genetic effects
    n_samples = []
    n_e_samples = []
    currentFeatureNumber = 0
    pvs = list()


    for feature_id in tqdm(feature_list):
        currentFeatureNumber+= 1

        snp_list = snp_feature_filter_df['snp_id'].loc[snp_feature_filter_df['feature'] == feature_id]
        snpQuery = bim.query('snp in @snp_list')

        if len(snpQuery) != 0:
            phenotype_ds = phenotype_df.loc[feature_id]
            contains_missing_samples = any(~np.isfinite(phenotype_ds))

            # import pdb; pdb.set_trace()

            if(contains_missing_samples):
                print ('Feature: ' + feature_id + ' contains missing data.')
                phenotype_ds.dropna(inplace = True)

            '''select indices for relevant individuals in genotype matrix
            These are not unique. NOT to be used to access phenotype/covariates data
            '''
            individual_ids = sample2individual_df.loc[phenotype_ds.index, 'iid'].values
            sample2individual_feature = sample2individual_df.loc[phenotype_ds.index]
            
            if(contains_missing_samples):
                tmp_unique_individuals = genetically_unique_individuals
                genetically_unique_individuals = utils.get_unique_genetic_samples(kinship_df.loc[individual_ids, individual_ids], relatedness_score);
            
            if phenotype_ds.empty or (len(genetically_unique_individuals) < minimum_test_samples) :
                print("Feature: " + feature_id + " not tested: not enough samples do QTL test.")
                # fail_qc_features.append(feature_id)
                if contains_missing_samples:
                    genetically_unique_individuals = tmp_unique_individuals
                continue
            elif np.var(phenotype_ds.values) == 0:
                print("Feature: " + feature_id + " has no variance in selected individuals.")
                # fail_qc_features.append(feature_id)
                if contains_missing_samples:
                    genetically_unique_individuals = tmp_unique_individuals
                continue

            
            #import pdb; pdb.set_trace()
            n_sample_t = (phenotype_ds.size)
            n_e_samples_t = (len(genetically_unique_individuals))
            print ('For feature: ' + str(currentFeatureNumber) + '/' +str(len(feature_list)) + ' (' + feature_id + '): ' + str(snpQuery.shape[0]) + ' SNPs need to be tested.\n Please stand by.')
            
            # import pdb; pdb.set_trace()


            for snpGroup in utils.chunker(snpQuery, blocksize):
                snp_idxs = snpGroup['i'].values
                snp_names = snpGroup['snp'].values
        

                #subset genotype matrix, we cannot subselect at the same time, do in two steps.
                snp_df = pd.DataFrame(data = bed[snp_idxs,:].compute().transpose(), index = fam.index, columns = snp_names)
                snp_df = snp_df.loc[individual_ids,:]
                
                
                #We could make use of relatedness when imputing.
                snp_matrix_DF = pd.DataFrame(fill_NaN.fit_transform(snp_df), index = snp_df.index, columns = snp_df.columns)
                snp_df = None

#                test if the environments, covariates, kinship, snp and phenotype are in the same order
                if ((all(snp_matrix_DF.index == kinship_df.loc[individual_ids,individual_ids].index) if kinship_df is not None else True) &\
                     (all(phenotype_ds.index == covariate_df.loc[sample2individual_feature['sample'],:].index)if covariate_df is not None else True)&\
                     all(phenotype_ds.index == environment_df.loc[sample2individual_feature['sample'],:].index)&\
                     all(snp_matrix_DF.index == sample2individual_feature.loc[phenotype_ds.index]['iid'])):
                    '''
                    if all lines are in order put in arrays the correct genotype and phenotype
                    x=a if cond1 else b <---> equivalent to if cond1: x=a else x=b;                 
                    better readability of the code
                    '''
                    kinship_mat = kinship_df.loc[individual_ids,individual_ids].values if kinship_df is not None else None
                    env =  environment_df.loc[sample2individual_feature['sample'],:].values
                    if covariate_df is not None:
                        cov_matrix =  covariate_df.loc[sample2individual_feature['sample'],:].values
                    else:
                        cov_matrix = None

                    phenotype = utils.force_normal_distribution(phenotype_ds.values, method = gaussianize_method) if gaussianize_method is not None else phenotype_ds.values
                else:
                    print ('There is an issue in mapping phenotypes and genotypes')
                    sys.exit()
                

                n_indep_snps = sum(limix.qc.indep_pairwise(snp_matrix_DF.values, window_size, window_size, threshold, verbose = True))
                print(n_indep_snps)
                # loop over SNPs to estimate individual effect sizes
                for snp in snp_names:
                        g = snp_matrix_DF.loc[:,snp].values
                        g = g.T.reshape(g.shape[0],1)
                        y = phenotype
                        y = y.T.reshape(y.shape[0],1)
                        W = cov_matrix
                        X = np.concatenate((W, g[:, np.newaxis]), axis = 1)
                        E = env
                        # fit null model
                        slmm_int = StructLMM2(y, W, E)
                        null = slmm_int.fit_null(g)
                        _p = slmm_int.score_2_dof(g)
                        # import pdb; pdb.set_trace()
                        pvs.append([snp, feature_id, _p, _p*n_indep_snps])

    
    pvs_df = pd.DataFrame(pvs, columns = ["snp_id","feature_id","p_value","feature_corrected_p_value"])
  
    if not selectionStart is None :
        # add if statements
        pvs_df.to_csv(output_dir + '/eqtl_gxe_pvalue_{}_{}_{}.txt'.format(chromosome, selectionStart, selectionEnd), sep = '\t', index = True)
    else :
        # add if statements
        pvs_df.to_csv(output_dir + '/eqtl_gxe_pvalue_{}.txt'.format(chromosome), sep = '\t', index = True)



if __name__=='__main__':

        #Variables
        chunkFile = '/nfs/leia/research/stegle/mjbonder/ChunkFiles/Ensembl_75_Limix_Annotation_FC_Gene_step10.txt'
        genotypeFile = '/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed'
        annotationFile = '/hps/nobackup/hipsci/scratch/singlecell_endodiff/data_processed/scQTLs/annos/combined_feature_id_annos.tsv'
        phenotypeFile = '/nfs/leia/research/stegle/acuomo/singlecell_endodiff/data/exploratory_analysis/input_files/full_sc_exprs_pheno.tsv'
        covariateFile = '/nfs/leia/research/stegle/acuomo/singlecell_endodiff/data/exploratory_analysis/input_files/full_sc_exprs_10pcs_covs.tsv'
        environmentFile = '/nfs/leia/research/stegle/acuomo/singlecell_endodiff/data/exploratory_analysis/input_files/full_sc_exprs_10pcs_envs.tsv'
        kinshipFile = '/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink-F/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.kinship'
        sampleMappingFile = '/nfs/leia/research/stegle/acuomo/singlecell_endodiff/data/exploratory_analysis/input_files/full_sc_exprs_samples.tsv'
        featureVariantFilter = '/nfs/leia/research/stegle/acuomo/singlecell_endodiff/data/exploratory_analysis/input_files/full_sc_exprs_merged_leads_top10_perstage.tsv'
        blockSize = '500'
        output_dir = '/nfs/leia/research/stegle/acuomo/singlecell_endodiff/data/pipeline_snakemakes/output_structlmm_alldays_10PCs/'
        
        run_structLMM2_int(phenotypeFile, annotationFile, environmentFile, genotypeFile, 
            output_dir, sample_mapping_filename = sampleMappingFile, snp_feature_filename = featureVariantFilter, 
            kinship_filename = kinshipFile, covariates_filename = covariateFile)



