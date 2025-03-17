#OBS: RODAR EM VENV

import pandas as pd
pd.options.display.max_columns = 300
import requests as req
import numpy as np

from google.cloud import bigquery
client = bigquery.Client()



##########################################################
### Importing the data                    ################
##########################################################


def import_sql(file_path):
  with open(file_path, 'r', encoding = 'utf-8') as file:
    sql_script = file.read()

    return sql_script

query_BPC = import_sql('query_BPC.sql')
query_agg_brand_inputs = import_sql('query_agg_brand_ue_tgt.sql')

df_bpc = client.query_and_wait(query_BPC).to_dataframe()
df_agg_brands_inputs = client.query_and_wait(query_agg_brand_inputs).to_dataframe()


##########################################################
### Checking data quality                 ################
##########################################################

#Checking for key columns completeness
df_bpc.describe() 
df_agg_brands_inputs.describe()


##########################################################
### Handling Missing data quality              ###########
##########################################################

df_bpc[['VISITS_COMPETITIVE','VISITS_MATCH']] = df_bpc[['VISITS_COMPETITIVE','VISITS_MATCH']].fillna(0)

##########################################################
### Adding new columns                         ###########
##########################################################

df_bpc['VISITS_COMPETITIVE_ESTIMATED'] = np.where(df_bpc['PRICE_MELI'] <= 1.01*df_bpc['COMP_PRICE_RIVAL'], df_bpc['VISITS_MATCH'],0)


##########################################################
### Finding the self representative AGG/Brands ###########
##########################################################

top50siteaggbrands= pd.DataFrame()

for site in df_bpc['SIT_SITE_ID'].unique():
  df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC']][df_bpc['SIT_SITE_ID']==site].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
  currenttop50keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(50)
  top50siteaggbrands = pd.concat([top50siteaggbrands,currenttop50keys])


top10verticalsaggbrands= pd.DataFrame()

for site in df_bpc['SIT_SITE_ID'].unique():
    for vertical in df_bpc['VERTICAL'].unique():
        df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC']][(df_bpc['SIT_SITE_ID']==site) & (df_bpc['VERTICAL']==vertical)].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
        currenttop10keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(10)
        top10verticalsaggbrands = pd.concat([top10verticalsaggbrands,currenttop10keys])

self_representative_agg_brands = pd.concat(
   [
      top50siteaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
      ,top10verticalsaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
    ]).drop_duplicates().sort_values(by = ['SIT_SITE_ID','VERTICAL']).reset_index(drop=True)


##########################################################
### STARTING A PROOF OF CONCEPT OF THE CALCULATOR ########
##########################################################

i = 60

example = self_representative_agg_brands.iloc[i]





df_bpc_filtered = df_bpc[ 
   (df_bpc['SIT_SITE_ID'] == example['SIT_SITE_ID']) & 
   (df_bpc['VERTICAL'] == example['VERTICAL']) & 
   (df_bpc['DOM_DOMAIN_AGG2'] == example['DOM_DOMAIN_AGG2']) & 
   (df_bpc['ITE_ATT_BRAND'] == example['ITE_ATT_BRAND']) 
   ]

df_agg_brands_inputs_filtered = df_agg_brands_inputs[ 
   (df_agg_brands_inputs['SIT_SITE_ID'] == example['SIT_SITE_ID']) & 
   (df_agg_brands_inputs['VERTICAL'] == example['VERTICAL']) & 
   (df_agg_brands_inputs['DOM_DOMAIN_AGG2'] == example['DOM_DOMAIN_AGG2']) & 
   (df_agg_brands_inputs['ITE_ATT_BRAND'] == example['ITE_ATT_BRAND']) 
   ]


newdf = pd.DataFrame()
newdf['BPC_original']= pd.Series(sum(df_bpc_filtered['VISITS_COMPETITIVE'])/sum(df_bpc_filtered['VISITS_MATCH']))
newdf['BPC_estimado']= pd.Series(sum(df_bpc_filtered['VISITS_COMPETITIVE_ESTIMATED'])/sum(df_bpc_filtered['VISITS_MATCH']))
newdf['BPC_tgt']= pd.Series(sum(df_agg_brands_inputs_filtered['TARGET_PRIORIZED']))

newdf['VM_lm']= pd.Series(sum(df_agg_brands_inputs_filtered['UE_MNG_VENDOR_MARGIN_AMT_LC_LM'])/sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_LM']))
newdf['VM_tgt']= pd.Series(sum(df_agg_brands_inputs_filtered['TGT_VENDOR_MARGIN_PERC_REV']))
newdf['VM_tgt']= pd.Series(sum(df_agg_brands_inputs_filtered['TGT_VENDOR_MARGIN_PERC_REV']))

newdf['VM_l6cm']= pd.Series(sum(df_agg_brands_inputs_filtered['UE_MNG_VENDOR_MARGIN_AMT_LC_L6CM'])/sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM']))
newdf['VC_l6cm']= pd.Series(sum(df_agg_brands_inputs_filtered['UE_MNG_VARIABLE_CONTRIBUTION_AMT_LC_L6CM'])/sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM']))
newdf['DC_l6cm']= pd.Series(sum(df_agg_brands_inputs_filtered['UE_MNG_DIRECT_CONTRIBUTION_AMT_LC_L6CM'])/sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM']))



