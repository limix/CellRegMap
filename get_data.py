import pandas as pd
import limix

def get_genotype_data(geno_prefix):
    bim,fam,bed = limix.io.plink.read(geno_prefix, verbose = False)
    fam.set_index('iid', inplace = True)
    return bim,fam,bed

'''
given a df of phenotype vectors df_y
and a sample mapping file
mapping genotype samples (e.g. individuals) to phenotype samples (e.g. cells)
returns genotype samples corresponding to the samples in y
'''
def map_samples(df_y, sample_mapping):
	df0 = pd.read_csv(sample_mapping, sep = " ")
	df0 = df0.set_index("phenotype_sample")
	return df0.loc[df_y.index, "genotype_sample"]

'''
given a df of phenotype vectors df_y
and a kinship matrix
returns kinship only containing samples included in phenotype
with the possibility of mapping samples first 
'''
def filter_kinship(K, df_y, sample_mapping = None):
    if sample_mapping is not None:
    	samples = map_samples(df_y, sample_mapping)
    else: samples = df_y.index
    K_filtered = K[samples,:][:,samples]
    return K_filtered

kinship_mat = kinship_df.loc[individual_ids,individual_ids].values if kinship_df is not None else None