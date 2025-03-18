#OBS: RODAR EM VENV

import pandas as pd
pd.options.display.max_columns = 300
pd.options.display.max_rows = 300

import requests as req
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
df_bpc[df_bpc['PPM_CALCULATED_FLOOR_PRICE'].isna()]

df_agg_brands_inputs.describe()


##########################################################
### Handling Missing data quality              ###########
##########################################################
df_bpc['PPM_CALCULATED_FLOOR_PRICE']=df_bpc['PPM_CALCULATED_FLOOR_PRICE'].fillna(0)
df_bpc['CCOGS']=df_bpc['CCOGS'].fillna(0)

df_bpc[['VISITS_COMPETITIVE','VISITS_MATCH']] = df_bpc[['VISITS_COMPETITIVE','VISITS_MATCH']].fillna(0)
df_bpc['FINANCIAL_COST'] = df_bpc['FINANCIAL_COST'].fillna(0)

##########################################################
### Adding new columns                         ###########
##########################################################

df_bpc['VISITS_COMPETITIVE_ESTIMATED'] = np.where(df_bpc['PRICE_MELI'] <= 1.01*df_bpc['COMP_PRICE_RIVAL'], df_bpc['VISITS_MATCH'],0)
df_bpc['PRICE_MELI2'] = np.where(df_bpc['PRICE_MELI'].isna(),df_bpc['TGMV_LC']/df_bpc['TSI'],df_bpc['PRICE_MELI'])
df_bpc['TGMV_LC_ESTIMATED']=df_bpc['PRICE_MELI2']*df_bpc['TSI']
# df_bpc['PRICE_TO_CHASE_0'] = np.where(df_bpc['COMP_PRICE_RIVAL'].isna(),df_bpc['PRICE_MELI2'], df_bpc[['COMP_PRICE_RIVAL','PRICE_MELI2']].values.min(1))
# df_bpc['PRICE_MELI_NEW_0'] = df_bpc[['PRICE_TO_CHASE_0','PPM_CALCULATED_FLOOR_PRICE']].values.max(1)
# df_bpc['TGMV_LC_NEW_0']=df_bpc['PRICE_MELI_NEW_0']*df_bpc['TSI']
# df_bpc['VISITS_COMPETITIVE_NEW_0'] = np.where(df_bpc['PRICE_MELI_NEW_0'] <= 1.01*df_bpc['COMP_PRICE_RIVAL'], df_bpc['VISITS_MATCH'],0)

# df_bpc['PPM_CALCULATED_FLOOR_PRICE_ESTIMATED'] = ((df_bpc['COST']-df_bpc['CCOGS'])*df_bpc['SIT_SITE_IVA'])/(1-df_bpc['PPM_PROFIT_FLOOR']/100-df_bpc['FINANCIAL_COST']/100)
# df_bpc[['COST','CCOGS','SIT_SITE_IVA','PPM_PROFIT_FLOOR','FINANCIAL_COST','PPM_CALCULATED_FLOOR_PRICE','PPM_CALCULATED_FLOOR_PRICE_ESTIMATED']][np.abs(df_bpc['PPM_CALCULATED_FLOOR_PRICE']-df_bpc['PPM_CALCULATED_FLOOR_PRICE'])==0]

##########################################################
### Finding the self representative AGG/Brands ###########
##########################################################

top50siteaggbrands= pd.DataFrame()
df_vm = df_agg_brands_inputs[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','UE_CON_TGMV_AMT_LC_L6CM']].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()

for site in df_bpc['SIT_SITE_ID'].unique():
  df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC','VISITS_MATCH']][df_bpc['SIT_SITE_ID']==site].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
  currenttop50keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(50)
  currenttop50keys = currenttop50keys.merge(df_vm, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ),right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ) )
  currenttop50keys=currenttop50keys[currenttop50keys['VISITS_MATCH']>0]
  currenttop50keys = currenttop50keys[currenttop50keys['UE_CON_TGMV_AMT_LC_L6CM'] > 0]
  top50siteaggbrands = pd.concat([top50siteaggbrands,currenttop50keys])


top10verticalsaggbrands= pd.DataFrame()

for site in df_bpc['SIT_SITE_ID'].unique():
    for vertical in df_bpc['VERTICAL'].unique():
        df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC','VISITS_MATCH']][(df_bpc['SIT_SITE_ID']==site) & (df_bpc['VERTICAL']==vertical)].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
        currenttop10keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(10)
        currenttop10keys = currenttop10keys.merge(df_vm, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ),right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ) )
        currenttop10keys=currenttop10keys[currenttop10keys['VISITS_MATCH']>0]
        currenttop10keys = currenttop10keys[currenttop10keys['UE_CON_TGMV_AMT_LC_L6CM'] > 0]
        top10verticalsaggbrands = pd.concat([top10verticalsaggbrands,currenttop10keys])

self_representative_agg_brands = pd.concat(
   [
      top50siteaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
      ,top10verticalsaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
    ]).drop_duplicates().sort_values(by = ['SIT_SITE_ID','VERTICAL']).reset_index(drop=True)


##########################################################
### STARTING A PROOF OF CONCEPT OF THE CALCULATOR ########
##########################################################

output_df = pd.DataFrame()

for i in range(0,len(self_representative_agg_brands)):
  example = self_representative_agg_brands.iloc[[i]]

  #Criar função a partir daqui


  df_bpc_filtered = df_bpc[ 
    (df_bpc['SIT_SITE_ID'] == example['SIT_SITE_ID'].iloc[0]) & 
    (df_bpc['VERTICAL'] == example['VERTICAL'].iloc[0]) & 
    (df_bpc['DOM_DOMAIN_AGG2'] == example['DOM_DOMAIN_AGG2'].iloc[0]) & 
    (df_bpc['ITE_ATT_BRAND'] == example['ITE_ATT_BRAND'].iloc[0]) 
    ]



  df_bpc_filtered['VISITS_COMPETITIVE_POTENTIAL']=np.where(df_bpc_filtered['PPM_CALCULATED_FLOOR_PRICE']<=df_bpc_filtered['COMP_PRICE_RIVAL'],df_bpc_filtered['VISITS_MATCH'],0)

  df_agg_brands_inputs_filtered = df_agg_brands_inputs[ 
    (df_agg_brands_inputs['SIT_SITE_ID'] == example['SIT_SITE_ID'].iloc[0]) & 
    (df_agg_brands_inputs['VERTICAL'] == example['VERTICAL'].iloc[0]) & 
    (df_agg_brands_inputs['DOM_DOMAIN_AGG2'] == example['DOM_DOMAIN_AGG2'].iloc[0]) & 
    (df_agg_brands_inputs['ITE_ATT_BRAND'] == example['ITE_ATT_BRAND'].iloc[0]) 
    ]



  newdf = example.copy()

  newdf['BPC_original']= sum(df_bpc_filtered['VISITS_COMPETITIVE'])/sum(df_bpc_filtered['VISITS_MATCH'])
  newdf['BPC_estimado']= sum(df_bpc_filtered['VISITS_COMPETITIVE_ESTIMATED'])/sum(df_bpc_filtered['VISITS_MATCH'])
  newdf['BPC_potencial']= sum(df_bpc_filtered['VISITS_COMPETITIVE_POTENTIAL'])/sum(df_bpc_filtered['VISITS_MATCH'])
  # newdf['BPC_NEW_0']= sum(df_bpc_filtered['VISITS_COMPETITIVE_NEW_0'])/sum(df_bpc_filtered['VISITS_MATCH'])

  newdf['BPC_tgt']= sum(df_agg_brands_inputs_filtered['TARGET_PRIORIZED'])

  newdf['VM_lm']= sum(df_agg_brands_inputs_filtered['UE_MNG_VENDOR_MARGIN_AMT_LC_LM'])/sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_LM'])
  newdf['VM_tgt']= sum(df_agg_brands_inputs_filtered['TGT_VENDOR_MARGIN_PERC_REV'])

  newdf['UE_MNG_REVENUE_GROSS_AMT_LC_LM'] = sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_LM'])
  newdf['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'] = sum(df_agg_brands_inputs_filtered['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'])
  newdf['UE_CON_CMV_AMT_LC_LM'] = sum(df_agg_brands_inputs_filtered['UE_CON_CMV_AMT_LC_LM'])
  newdf['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM'] = sum(df_agg_brands_inputs_filtered['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM'])
  newdf['UE_CON_CONTRACOGS_AMT_LC_LM'] = sum(df_agg_brands_inputs_filtered['UE_CON_CONTRACOGS_AMT_LC_LM'])




  newdf['VM_l6cm']= sum(df_agg_brands_inputs_filtered['UE_MNG_VENDOR_MARGIN_AMT_LC_L6CM'])/sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'])
  newdf['VC_l6cm']= sum(df_agg_brands_inputs_filtered['UE_MNG_VARIABLE_CONTRIBUTION_AMT_LC_L6CM'])/sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'])
  newdf['DC_l6cm']= sum(df_agg_brands_inputs_filtered['UE_MNG_DIRECT_CONTRIBUTION_AMT_LC_L6CM'])/sum(df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'])

  newdf['TGMV_LC']= sum(df_bpc_filtered['TGMV_LC'])
  newdf['TGMV_LC_ESTIMATED']= sum(df_bpc_filtered['TGMV_LC_ESTIMATED'])

  newdf['TSI'] = sum(df_bpc_filtered['TSI'])
  # newdf['TGMV_LC_NEW_0']= sum(df_bpc_filtered['TGMV_LC_NEW_0'])

  grid_df = pd.DataFrame()

  for new_ppm in range(-10, 55 + 1):

    df_bpc_filtered['PRICE_TO_CHASE_X'] = np.where(df_bpc_filtered['COMP_PRICE_RIVAL'].isna(),df_bpc_filtered['PRICE_MELI2'], df_bpc_filtered[['COMP_PRICE_RIVAL','PRICE_MELI2']].values.min(1))
    df_bpc_filtered['PPM_CALCULATED_FLOOR_PRICE_X'] = ((df_bpc['COST']-df_bpc['CCOGS'])*df_bpc['SIT_SITE_IVA'])/(1-new_ppm/100-df_bpc['FINANCIAL_COST']/100)
    df_bpc_filtered['PRICE_MELI_NEW_X'] = df_bpc_filtered[['PRICE_TO_CHASE_X','PPM_CALCULATED_FLOOR_PRICE_X']].values.max(1)
    df_bpc_filtered['TSI_NEW_X'] =  df_bpc_filtered['TSI']*((1+df_bpc_filtered['B_EFECTIVO']/100)**(100*((df_bpc_filtered['PRICE_MELI_NEW_X'] - df_bpc_filtered['PRICE_MELI2'])/df_bpc_filtered['PRICE_MELI2'])))
    #df_bpc_filtered[['TSI','TSI_NEW_X ','B_EFECTIVO','PRICE_MELI2','PRICE_MELI_NEW_X']][df_bpc_filtered['TSI']!=df_bpc_filtered['TSI_NEW_X']]
    df_bpc_filtered['TGMV_LC_NEW_X']=df_bpc_filtered['PRICE_MELI_NEW_X']*df_bpc_filtered['TSI_NEW_X']
    df_bpc_filtered['VISITS_COMPETITIVE_NEW_X'] = np.where(df_bpc_filtered['PRICE_MELI_NEW_X'] <= 1.01*df_bpc_filtered['COMP_PRICE_RIVAL'], df_bpc_filtered['VISITS_MATCH'],0)
    pd.Series(sum(df_bpc_filtered['VISITS_COMPETITIVE_NEW_X'])/sum(df_bpc_filtered['VISITS_MATCH']))
    pd.Series(sum(df_bpc_filtered['TSI']))
    pd.Series(sum(df_bpc_filtered['TSI_NEW_X']))

    new_row =  newdf.copy()
    new_row['NEW_PPM']= new_ppm
    new_row['TGMV_LC_NEW_X'] = sum(df_bpc_filtered['TGMV_LC_NEW_X'])
    new_row['TSI_NEW_X'] = sum(df_bpc_filtered['TSI_NEW_X'])
    new_row['BPC_NEW_X']= sum(df_bpc_filtered['VISITS_COMPETITIVE_NEW_X'])/sum(df_bpc_filtered['VISITS_MATCH'])


  # USAR DADOS LM?
    new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM'] * new_row['TGMV_LC_NEW_X']/new_row['TGMV_LC_ESTIMATED']
    new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'] * 1
    new_row['UE_CON_CMV_AMT_LC_LM_NEW_X'] =  new_row['UE_CON_CMV_AMT_LC_LM'] * new_row['TSI_NEW_X']/new_row['TSI']
    new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM']* new_row['TSI_NEW_X']/new_row['TSI']
    new_row['UE_CON_CONTRACOGS_AMT_LC_LM_NEW_X'] = new_row['UE_CON_CONTRACOGS_AMT_LC_LM'] * 1
    new_row['VM_LM_NEW_X_PERC_REV'] = (new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X'] + new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM_NEW_X'] + new_row['UE_CON_CMV_AMT_LC_LM_NEW_X'] + new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM_NEW_X'] + new_row['UE_CON_CONTRACOGS_AMT_LC_LM_NEW_X'])/(new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X']) 
  
    grid_df = pd.concat([grid_df,new_row])

  final_row = grid_df[grid_df['BPC_NEW_X'] >= grid_df['BPC_tgt']].tail(1)
  
    #Create grid_df from newdf and update at each iteration

  output_df = pd.concat([output_df,final_row])

  print(i)

output_df