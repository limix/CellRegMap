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

def run_structLMM_QTL_analysis_load_intersect_phenotype_environments_covariates_kinship_sample_mapping\
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
    if(sample2individual_df.shape[0] < minimum_test_samples):
        print("Not enough samples with both genotype & phenotype data.")
        sys.exit()

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





