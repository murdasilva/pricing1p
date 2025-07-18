#OBS: RODAR EM VENV

import pandas as pd
import pandas_gbq
pd.options.display.max_columns = 300
pd.options.display.max_rows = 70

import requests as req
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from google.cloud import bigquery
client = bigquery.Client()

import datetime
today = datetime.date.today()
formatted_date = today.strftime("%Y-%m-%d") 

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
df_bpc['TGMV_LC']=df_bpc['TGMV_LC'].fillna(0)
df_bpc['TSI']=df_bpc['TSI'].fillna(0)



df_bpc[['VISITS_COMPETITIVE','VISITS_MATCH']] = df_bpc[['VISITS_COMPETITIVE','VISITS_MATCH']].fillna(0)
df_bpc['FINANCIAL_COST'] = df_bpc['FINANCIAL_COST'].fillna(0)


df_agg_brands_inputs['UE_CON_TGMV_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_CON_TGMV_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_VENDOR_MARGIN_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_VENDOR_MARGIN_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_VENDOR_MARGIN_ADJ_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_VENDOR_MARGIN_ADJ_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_VARIABLE_CONTRIBUTION_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_VARIABLE_CONTRIBUTION_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_VARIABLE_CONTRIBUTION_ADJ_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_VARIABLE_CONTRIBUTION_ADJ_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_DIRECT_CONTRIBUTION_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_DIRECT_CONTRIBUTION_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_DIRECT_CONTRIBUITION_ADJ_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_DIRECT_CONTRIBUITION_ADJ_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_CON_CMV_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_CON_CMV_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_CON_CONTRACOGS_AMT_LC_L6CM'] = df_agg_brands_inputs['UE_CON_CONTRACOGS_AMT_LC_L6CM'].fillna(0)
df_agg_brands_inputs['UE_CON_TGMV_AMT_LC_LM'] = df_agg_brands_inputs['UE_CON_TGMV_AMT_LC_LM'].fillna(0)
df_agg_brands_inputs['UE_MNG_VENDOR_MARGIN_AMT_LC_LM'] = df_agg_brands_inputs['UE_MNG_VENDOR_MARGIN_AMT_LC_LM'].fillna(0)
df_agg_brands_inputs['UE_MNG_REVENUE_GROSS_AMT_LC_LM'] = df_agg_brands_inputs['UE_MNG_REVENUE_GROSS_AMT_LC_LM'].fillna(0)
df_agg_brands_inputs['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'] = df_agg_brands_inputs['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'].fillna(0)
df_agg_brands_inputs['UE_CON_CMV_AMT_LC_LM'] = df_agg_brands_inputs['UE_CON_CMV_AMT_LC_LM'].fillna(0)
df_agg_brands_inputs['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM'] = df_agg_brands_inputs['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM'].fillna(0)
df_agg_brands_inputs['UE_CON_CONTRACOGS_AMT_LC_LM'] = df_agg_brands_inputs['UE_CON_CONTRACOGS_AMT_LC_LM'].fillna(0)
df_agg_brands_inputs['TGMV_USD_LM'] = df_agg_brands_inputs['TGMV_USD_LM'].fillna(0)
df_agg_brands_inputs['SHARE_TGMV_USD_LM'] = df_agg_brands_inputs['SHARE_TGMV_USD_LM'].fillna(0)




##########################################################
### Adding new columns                         ###########
##########################################################

df_bpc['VISITS_COMPETITIVE_ESTIMATED'] = np.where(df_bpc['PRICE_MELI'] <= 1.01*df_bpc['COMP_PRICE_RIVAL'], df_bpc['VISITS_MATCH'],0)
df_bpc['PRICE_MELI2'] = np.where(df_bpc['PRICE_MELI'].isna(),df_bpc['TGMV_LC']/df_bpc['TSI'],df_bpc['PRICE_MELI'])
df_bpc['PRICE_TO_CHASE'] = np.where(df_bpc['COMP_PRICE_RIVAL'].isna(),df_bpc['PRICE_MELI2'], df_bpc[['COMP_PRICE_RIVAL','PRICE_MELI2']].values.min(1))
df_bpc['PPM_CALCULATED_FLOOR_PRICE_ESTIMATED'] = ((df_bpc['COST']-df_bpc['CCOGS'])*df_bpc['SIT_SITE_IVA'])/(1-df_bpc['PPM_PROFIT_FLOOR']/100-df_bpc['FINANCIAL_COST']/100).fillna(0)


df_bpc['EFFECTIVE_FLOOR_PRICE'] = df_bpc['PPM_CALCULATED_FLOOR_PRICE_ESTIMATED'].copy().fillna(0)
df_bpc['EFFECTIVE_FLOOR_PRICE'][ (df_bpc['PL1P_PRICING_CURRENT_WINNING_STRATEGY'] == 'DEAL') | (df_bpc['PL1P_PRICING_CURRENT_WINNING_STRATEGY'] == 'PROMO') | (df_bpc['PL1P_PRICING_CURRENT_WINNING_STRATEGY'] == 'MARKDOWN')] = df_bpc[['PPM_CALCULATED_FLOOR_PRICE_ESTIMATED','PRICE_MELI2']].min(axis=1)
df_bpc['PRICE_MELI_NEW'] = df_bpc[['PRICE_TO_CHASE','EFFECTIVE_FLOOR_PRICE']].values.max(1)
df_bpc['TGMV_LC_ESTIMATED']=df_bpc['PRICE_MELI_NEW']*df_bpc['TSI']
df_bpc['VISITS_COMPETITIVE_POTENTIAL']=np.where(df_bpc[['PPM_CALCULATED_FLOOR_PRICE','PRICE_MELI_NEW']].min(axis=1)<=1.01*df_bpc['COMP_PRICE_RIVAL'],df_bpc['VISITS_MATCH'],0)

##########################################################
### Finding the self representative AGG/Brands ###########
##########################################################

top500allaggbrands = pd.DataFrame()
top500allaggbrands = df_agg_brands_inputs[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_USD_LM']].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index().sort_values(by='TGMV_USD_LM',ascending =False).head(500).reset_index(drop=True)

# top50siteaggbrands= pd.DataFrame()
df_vm = df_agg_brands_inputs[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','UE_CON_TGMV_AMT_LC_L6CM','UE_CON_TGMV_AMT_LC_LM']].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()

# for site in df_bpc['SIT_SITE_ID'].unique():
#   df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC','VISITS_MATCH']][df_bpc['SIT_SITE_ID']==site].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
#   currenttop50keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(50)
#   currenttop50keys = currenttop50keys.merge(df_vm, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ),right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ) )
#   currenttop50keys=currenttop50keys[currenttop50keys['VISITS_MATCH']>0]
#   currenttop50keys = currenttop50keys[currenttop50keys['UE_CON_TGMV_AMT_LC_L6CM'] > 0]
#   currenttop50keys = currenttop50keys[currenttop50keys['UE_CON_TGMV_AMT_LC_LM'] > 0]
#   top50siteaggbrands = pd.concat([top50siteaggbrands,currenttop50keys])


top20siteaggbrands= pd.DataFrame()
for site in df_bpc['SIT_SITE_ID'].unique():
  df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC','VISITS_MATCH']][df_bpc['SIT_SITE_ID']==site].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
  currenttop20keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(20)
  currenttop20keys = currenttop20keys.merge(df_vm, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ),right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ) )
  currenttop20keys=currenttop20keys[currenttop20keys['VISITS_MATCH']>0]
  currenttop20keys = currenttop20keys[currenttop20keys['UE_CON_TGMV_AMT_LC_L6CM'] > 0]
  currenttop20keys = currenttop20keys[currenttop20keys['UE_CON_TGMV_AMT_LC_LM'] > 0]
  top20siteaggbrands = pd.concat([top20siteaggbrands,currenttop20keys])

top20siteaggbrands = top20siteaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
top20siteaggbrands['FLAG_top20_AGGBRAND'] = 1

# top10verticalsaggbrands= pd.DataFrame()

# for site in df_bpc['SIT_SITE_ID'].unique():
#     for vertical in df_bpc['VERTICAL'].unique():
#         df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC','VISITS_MATCH']][(df_bpc['SIT_SITE_ID']==site) & (df_bpc['VERTICAL']==vertical)].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
#         currenttop10keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(10)
#         currenttop10keys = currenttop10keys.merge(df_vm, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ),right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ) )
#         currenttop10keys=currenttop10keys[currenttop10keys['VISITS_MATCH']>0]
#         currenttop10keys = currenttop10keys[currenttop10keys['UE_CON_TGMV_AMT_LC_L6CM'] > 0]
#         currenttop10keys = currenttop10keys[currenttop10keys['UE_CON_TGMV_AMT_LC_LM'] > 0]
#         top10verticalsaggbrands = pd.concat([top10verticalsaggbrands,currenttop10keys])



##########################################################
### Finding the self representative AGGs      ############ 
##########################################################
tsi_threshold = 10
df_vm_agg = df_agg_brands_inputs[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','UE_CON_TGMV_AMT_LC_L6CM','UE_CON_TGMV_AMT_LC_LM']].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2']).sum().reset_index()

aggkeys = pd.DataFrame()
# AGGs where TSI > Threshold
for site in df_bpc['SIT_SITE_ID'].unique():
  df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','TGMV_LC','TSI','VISITS_MATCH']][df_bpc['SIT_SITE_ID']==site].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2']).sum().reset_index()
  df_tgmv = df_tgmv.merge(df_vm_agg, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2' ),right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2') )
  df_tgmv=df_tgmv[df_tgmv['VISITS_MATCH']>0]
  df_tgmv = df_tgmv[df_tgmv['UE_CON_TGMV_AMT_LC_L6CM'] > 0]
  df_tgmv = df_tgmv[df_tgmv['UE_CON_TGMV_AMT_LC_LM'] > 0]
  df_tgmv = df_tgmv[df_tgmv['TSI'] > tsi_threshold]
  aggkeys = pd.concat([aggkeys,df_tgmv])

aggkeys['ITE_ATT_BRAND'] = 'ALL_BRANDS'


self_representative_agg_brands = pd.concat(
   [
      top500allaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
      #,top10verticalsaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
      ,aggkeys[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
    ]).drop_duplicates().sort_values(by = ['SIT_SITE_ID','VERTICAL']).reset_index(drop=True)


##########################################################
### STARTING A PROOF OF CONCEPT OF THE CALCULATOR ########
##########################################################

def mask_function(df,example_df):
  if example_df['ITE_ATT_BRAND'].iloc[0] == 'ALL_BRANDS':
    mask =    ( 
      (df['SIT_SITE_ID'] == example_df['SIT_SITE_ID'].iloc[0]) & 
      (df['VERTICAL'] == example_df['VERTICAL'].iloc[0]) & 
      (df['DOM_DOMAIN_AGG2'] == example_df['DOM_DOMAIN_AGG2'].iloc[0]) #& 
      # (df_bpc['ITE_ATT_BRAND'] == example_df['ITE_ATT_BRAND'].iloc[0]) 
    )
  else:
    mask =    ( 
      (df['SIT_SITE_ID'] == example_df['SIT_SITE_ID'].iloc[0]) & 
      (df['VERTICAL'] == example_df['VERTICAL'].iloc[0]) & 
      (df['DOM_DOMAIN_AGG2'] == example_df['DOM_DOMAIN_AGG2'].iloc[0]) & 
      (df['ITE_ATT_BRAND'] == example_df['ITE_ATT_BRAND'].iloc[0]) 
    )
  return mask

###################################################################################
def bpc_calculator(bpc_df, agg_brands_df, example_df, min_ppm = -15, max_ppm = 55):
  
  # Defining the masks
  mask_bpc = mask_function(bpc_df,example_df)
  mask_inputs = mask_function(agg_brands_df,example_df)

  #Filtering the dfs
  df_bpc_filtered = bpc_df[ mask_bpc ]
  df_agg_brands_inputs_filtered = agg_brands_df[ mask_inputs ]

  # Calculating new columns before iterating the PPM values

  newdf = example_df.copy()
  newdf['BPC_original']= df_bpc_filtered['VISITS_COMPETITIVE'].sum()/df_bpc_filtered['VISITS_MATCH'].sum()
  newdf['BPC_ABC_original']= df_agg_brands_inputs_filtered['VISITS_COMPETITIVE_ABC'].sum()/df_agg_brands_inputs_filtered['VISITS_MATCH_ABC'].sum()
  newdf['BPC_estimado']= df_bpc_filtered['VISITS_COMPETITIVE_ESTIMATED'].sum()/df_bpc_filtered['VISITS_MATCH'].sum()
  newdf['BPC_potencial']= df_bpc_filtered['VISITS_COMPETITIVE_POTENTIAL'].sum()/df_bpc_filtered['VISITS_MATCH'].sum()
  newdf['BPC_tgt']= df_agg_brands_inputs_filtered['TARGET_PRIORIZED'].mean()
  newdf['VISITS_MATCH'] = df_bpc_filtered['VISITS_MATCH'].sum()

  newdf['VM_lm']= df_agg_brands_inputs_filtered['UE_MNG_VENDOR_MARGIN_AMT_LC_LM'].sum()/df_agg_brands_inputs_filtered['UE_CON_TGMV_AMT_LC_LM'].sum()
  newdf['VM_tgt']= df_agg_brands_inputs_filtered['TGT_VENDOR_MARGIN_PERC_TGMV'].mean()

  newdf['UE_CON_TGMV_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_CON_TGMV_AMT_LC_LM'].sum()
  newdf['UE_MNG_REVENUE_GROSS_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_LM'].sum()
  newdf['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'].sum()
  newdf['UE_CON_CMV_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_CON_CMV_AMT_LC_LM'].sum()
  newdf['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM'].sum()
  newdf['UE_CON_CONTRACOGS_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_CON_CONTRACOGS_AMT_LC_LM'].sum()

  # newdf['VM_l6cm']= df_agg_brands_inputs_filtered['UE_MNG_VENDOR_MARGIN_AMT_LC_L6CM'].sum()/df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'].sum()
  # newdf['VC_l6cm']= df_agg_brands_inputs_filtered['UE_MNG_VARIABLE_CONTRIBUTION_AMT_LC_L6CM'].sum()/df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'].sum()
  # newdf['DC_l6cm']= df_agg_brands_inputs_filtered['UE_MNG_DIRECT_CONTRIBUTION_AMT_LC_L6CM'].sum()/df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'].sum()

  newdf['TGMV_LC']= df_bpc_filtered['TGMV_LC'].sum()
  newdf['TGMV_LC_ESTIMATED']= df_bpc_filtered['TGMV_LC_ESTIMATED'].sum()
  newdf['TSI'] = df_bpc_filtered['TSI'].sum()
  newdf['DC_PERC_L6M']= df_agg_brands_inputs_filtered['UE_MNG_DIRECT_CONTRIBUITION_ADJ_AMT_LC_L6CM'].sum()/df_agg_brands_inputs_filtered['UE_CON_TGMV_AMT_LC_L6CM'].sum()

  grid_df = pd.DataFrame()

  # Starting the grid search by loop
  for new_ppm in range(min_ppm, max_ppm + 1):

    df_bpc_filtered['PRICE_TO_CHASE_X'] = np.where(df_bpc_filtered['COMP_PRICE_RIVAL'].isna(),df_bpc_filtered['PRICE_MELI2'], df_bpc_filtered[['COMP_PRICE_RIVAL','PRICE_MELI2']].values.min(1))
    df_bpc_filtered['PPM_CALCULATED_FLOOR_PRICE_X'] = ((df_bpc_filtered['COST']-df_bpc_filtered['CCOGS'])*df_bpc_filtered['SIT_SITE_IVA'])/(1-new_ppm/100-df_bpc_filtered['FINANCIAL_COST']/100)
    df_bpc_filtered['PPM_CALCULATED_FLOOR_PRICE_X'] = df_bpc_filtered['PPM_CALCULATED_FLOOR_PRICE_X'].fillna(0)
    df_bpc_filtered['EFFECTIVE_FLOOR_PRICE_X'] = df_bpc_filtered['PPM_CALCULATED_FLOOR_PRICE_X'].copy()
    df_bpc_filtered['EFFECTIVE_FLOOR_PRICE_X'][ (df_bpc_filtered['PL1P_PRICING_CURRENT_WINNING_STRATEGY'] == 'DEAL') | (df_bpc_filtered['PL1P_PRICING_CURRENT_WINNING_STRATEGY'] == 'PROMO') | (df_bpc_filtered['PL1P_PRICING_CURRENT_WINNING_STRATEGY'] == 'MARKDOWN')] = df_bpc_filtered[['PPM_CALCULATED_FLOOR_PRICE_X','PRICE_MELI2']].min(axis=1)
    df_bpc_filtered['PRICE_MELI_NEW_X'] = df_bpc_filtered[['PRICE_TO_CHASE_X','EFFECTIVE_FLOOR_PRICE_X']].values.max(1)
    # df_bpc_filtered['TSI_NEW_X'] =  df_bpc_filtered['TSI']
    df_bpc_filtered['TSI_NEW_X'] =  df_bpc_filtered['TSI']*((1+df_bpc_filtered['B_EFECTIVO']/100)**(100*((df_bpc_filtered['PRICE_MELI_NEW_X'] - df_bpc_filtered['PRICE_MELI2'])/df_bpc_filtered['PRICE_MELI2'])))
    #df_bpc_filtered[['TSI','TSI_NEW_X ','B_EFECTIVO','PRICE_MELI2','PRICE_MELI_NEW_X']][df_bpc_filtered['TSI']!=df_bpc_filtered['TSI_NEW_X']]
    df_bpc_filtered['TGMV_LC_NEW_X']=df_bpc_filtered['PRICE_MELI_NEW_X']*df_bpc_filtered['TSI_NEW_X']
    df_bpc_filtered['VISITS_COMPETITIVE_NEW_X'] = np.where(df_bpc_filtered['PRICE_MELI_NEW_X'] <= 1.01*df_bpc_filtered['COMP_PRICE_RIVAL'], df_bpc_filtered['VISITS_MATCH'],0)


    new_row =  newdf.copy()
    new_row['NEW_PPM']= new_ppm
    new_row['TGMV_LC_NEW_X'] = df_bpc_filtered['TGMV_LC_NEW_X'].sum()
    new_row['TSI_NEW_X'] = df_bpc_filtered['TSI_NEW_X'].sum()
    new_row['BPC_NEW_X']= df_bpc_filtered['VISITS_COMPETITIVE_NEW_X'].sum()/df_bpc_filtered['VISITS_MATCH'].sum()
    new_row['TSI_VALUED'] = (df_bpc_filtered['TSI']*df_bpc_filtered['COST']).sum()
    new_row['TSI_NEW_X_VALUED'] = (df_bpc_filtered['TSI_NEW_X']*df_bpc_filtered['COST']).sum()

    new_row['UE_CON_TGMV_AMT_LC_LM_NEW_X'] = new_row['UE_CON_TGMV_AMT_LC_LM'] * new_row['TGMV_LC_NEW_X']/new_row['TGMV_LC_ESTIMATED']
    new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM'] * new_row['TGMV_LC_NEW_X']/new_row['TGMV_LC_ESTIMATED']
    new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'] * 1
    new_row['UE_CON_CMV_AMT_LC_LM_NEW_X'] =  new_row['UE_CON_CMV_AMT_LC_LM'] * new_row['TSI_NEW_X_VALUED']/new_row['TSI_VALUED']
    new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM']* new_row['TSI_NEW_X_VALUED']/new_row['TSI_VALUED']
    new_row['UE_CON_CONTRACOGS_AMT_LC_LM_NEW_X'] = new_row['UE_CON_CONTRACOGS_AMT_LC_LM'] * 1
    
    new_row['VM_LM_NEW_X_PERC_REV'] = (new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X'] + new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM_NEW_X'] + new_row['UE_CON_CMV_AMT_LC_LM_NEW_X'] + new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM_NEW_X'] + new_row['UE_CON_CONTRACOGS_AMT_LC_LM_NEW_X'])/(new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X']) 
    new_row['VM_LM_NEW_X_PERC_TGMV'] = (new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X'] + new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM_NEW_X'] + new_row['UE_CON_CMV_AMT_LC_LM_NEW_X'] + new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM_NEW_X'] + new_row['UE_CON_CONTRACOGS_AMT_LC_LM_NEW_X'])/(new_row['UE_CON_TGMV_AMT_LC_LM_NEW_X']) 

    new_row['CURRENT_PPM_FLOOR'] = round((df_bpc_filtered[df_bpc_filtered['PPM_PROFIT_FLOOR'].notna()]['PPM_PROFIT_FLOOR']*df_bpc_filtered[df_bpc_filtered['PPM_PROFIT_FLOOR'].notna()]['VISITS_MATCH']).sum()/df_bpc_filtered[df_bpc_filtered['PPM_PROFIT_FLOOR'].notna()]['VISITS_MATCH'].sum(),0)
    

    grid_df = pd.concat([grid_df,new_row])
  
  ppm_that_yields_both_tgts = grid_df[(grid_df['BPC_NEW_X'] >= grid_df['BPC_tgt']) & (grid_df['VM_LM_NEW_X_PERC_TGMV'] >= grid_df['VM_tgt']) ].tail(5).head(1) # PPM that allows for both targets  (.tail(5) garante que nesses casos ficamos perto do VM_target)
  ppm_that_yields_bpc_tgt = grid_df[grid_df['BPC_NEW_X'] >= grid_df['BPC_tgt']].tail(1) # Greatest PPM that allows for BPC tgt
  ppm_min = grid_df.head(1) #Smallest allowed PPM 

  final_df = pd.concat([ppm_that_yields_both_tgts,ppm_that_yields_bpc_tgt,ppm_min])
  final_row = final_df.iloc[[0]]

  return final_row , grid_df

##################################################################
output_df = pd.DataFrame()
all_grids_df = pd.DataFrame()

for i in range(0,len(self_representative_agg_brands)):
 
  example = self_representative_agg_brands.iloc[[i]]
  final_row, grid_df = bpc_calculator(df_bpc, df_agg_brands_inputs, example, min_ppm = -10, max_ppm = 55)
  output_df = pd.concat([output_df,final_row])
  all_grids_df = pd.concat([all_grids_df,grid_df])
  print(i)


## SAving the allgrids_df to a Bigquery table

project_id = "meli-bi-data"

table_id = 'SBOX_PRICING1P.TEMP_ALL_GRIDS_DF'


pandas_gbq.to_gbq(all_grids_df, table_id, project_id=project_id,if_exists='replace')


# Adding new columns to output_df

output_df['BUCKET'] = np.nan
output_df['BUCKET'][(output_df['VM_lm']>=output_df['VM_tgt']) & (output_df[['BPC_original','BPC_potencial']].max(axis=1)>=output_df['BPC_tgt'])  ] = '1. MANTENER'
output_df['BUCKET'][(output_df['VM_lm']<output_df['VM_tgt'])  & (output_df[['BPC_original','BPC_potencial']].max(axis=1)>=output_df['BPC_tgt']) & (output_df['BPC_NEW_X']>=output_df['BPC_tgt']) & (output_df['VM_LM_NEW_X_PERC_TGMV']>=output_df['VM_tgt'])] = '4. RENTABILIZAR A TARGET'
output_df['BUCKET'][(output_df['VM_lm']<output_df['VM_tgt'])  & (output_df[['BPC_original','BPC_potencial']].max(axis=1)>=output_df['BPC_tgt']) & (output_df['BPC_NEW_X']>=output_df['BPC_tgt']) & (output_df['VM_LM_NEW_X_PERC_TGMV']<output_df['VM_tgt'])] = '5. RENTABILIZAR PARCIALMENTE'
output_df['BUCKET'][(output_df[['BPC_original','BPC_potencial']].max(axis=1)<output_df['BPC_tgt']) & (output_df['VM_LM_NEW_X_PERC_TGMV']>=output_df['VM_tgt']) & (output_df['BPC_NEW_X']>=output_df['BPC_tgt']) ] = '2. COMPETITIVIZAR'
output_df['BUCKET'][(output_df[['BPC_original','BPC_potencial']].max(axis=1)<output_df['BPC_tgt']) & (output_df['VM_LM_NEW_X_PERC_TGMV']<output_df['VM_tgt']) & (output_df['BPC_NEW_X']>=output_df['BPC_tgt']) ] = '3. INVERTIR'
output_df['BUCKET'][(output_df['BPC_NEW_X']<output_df['BPC_tgt']) ] = '6. REVISAR'
output_df['BUCKET'] = output_df['BUCKET'].fillna('X. ERROR')

#Adicionar coluna de piso final
output_df['FINAL_PPM_FLOOR'] = np.nan
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '1. MANTENER']= output_df['CURRENT_PPM_FLOOR']
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '4. RENTABILIZAR A TARGET']= output_df[['NEW_PPM','CURRENT_PPM_FLOOR']].max(axis=1)
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '5. RENTABILIZAR PARCIALMENTE']= output_df[['NEW_PPM','CURRENT_PPM_FLOOR']].max(axis=1)
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '2. COMPETITIVIZAR']= output_df[['NEW_PPM','CURRENT_PPM_FLOOR']].min(axis=1)
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '3. INVERTIR']= output_df[['NEW_PPM','CURRENT_PPM_FLOOR']].min(axis=1)
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '6. REVISAR']= output_df['CURRENT_PPM_FLOOR']

#Adicionar Flag de Top 10
output_df = output_df.merge(top20siteaggbrands, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND'), right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND'))
output_df['FLAG_top20_AGGBRAND'] = output_df['FLAG_top20_AGGBRAND'].fillna(0)

#Adicionar coluna de Governança
output_df['GOVERNANCE'] = 'ERROR'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '1. MANTENER') ]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '4. RENTABILIZAR A TARGET') & (output_df['FLAG_top20_AGGBRAND'] == 0)]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '4. RENTABILIZAR A TARGET') & (output_df['FLAG_top20_AGGBRAND'] == 1)]= 'B. MANAGER/DIRECTOR'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '5. RENTABILIZAR PARCIALMENTE')& (output_df['FLAG_top20_AGGBRAND'] == 0)]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '5. RENTABILIZAR PARCIALMENTE')& (output_df['FLAG_top20_AGGBRAND'] == 1)]= 'B. MANAGER/DIRECTOR'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '6. REVISAR')]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR']>=0) & (output_df['BUCKET'] == '2. COMPETITIVIZAR') & (output_df['FLAG_top20_AGGBRAND'] == 0)]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR']>=0) & (output_df['BUCKET'] == '2. COMPETITIVIZAR') & (output_df['FLAG_top20_AGGBRAND'] == 1) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_TGMV'] <= output_df['DC_PERC_L6M'] )]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR']>=0) & (output_df['BUCKET'] == '3. INVERTIR') & (output_df['FLAG_top20_AGGBRAND'] == 0)]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR']>=0) & (output_df['BUCKET'] == '3. INVERTIR') & (output_df['FLAG_top20_AGGBRAND'] == 1) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_TGMV'] <= output_df['DC_PERC_L6M'] )]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR']>=0) & (output_df['BUCKET'] == '2. COMPETITIVIZAR') & (output_df['FLAG_top20_AGGBRAND'] == 1) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_TGMV'] > output_df['DC_PERC_L6M'] )]= 'B. MANAGER/DIRECTOR'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR']>=0) & (output_df['BUCKET'] == '3. INVERTIR') & (output_df['FLAG_top20_AGGBRAND'] == 1) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_TGMV'] > output_df['DC_PERC_L6M'] )]= 'B. MANAGER/DIRECTOR'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR']<0) & (output_df['BUCKET'] == '2. COMPETITIVIZAR') ]= 'C. DIRECTOR/VP'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR']<0) & (output_df['BUCKET'] == '3. INVERTIR')]= 'C. DIRECTOR/VP'

output_df.describe()

pd.crosstab(output_df['BUCKET'][output_df['ITE_ATT_BRAND']!='ALL_BRANDS'],output_df['GOVERNANCE'],margins = True)
pd.crosstab(output_df['BUCKET'][output_df['ITE_ATT_BRAND']=='ALL_BRANDS'],output_df['GOVERNANCE'],margins = True)

# Visitas por top AGG2/BRANDS
pd.pivot_table(output_df[output_df['ITE_ATT_BRAND']!='ALL_BRANDS'], values=['VISITS_MATCH'], index=['BUCKET'], columns=['GOVERNANCE'], aggfunc=np.sum, margins=True,fill_value = 0)
# Visitas por AGG2
pd.pivot_table(output_df[output_df['ITE_ATT_BRAND']=='ALL_BRANDS'], values=['VISITS_MATCH'], index=['BUCKET'], columns=['GOVERNANCE'], aggfunc=np.sum, margins=True,fill_value = 0)


# # CHECAR SE TEM CASOS ASSIM
# output_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','NEW_PPM','BPC_NEW_X','VM_LM_NEW_X_PERC_TGMV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE']][((output_df['BUCKET']=='5. RENTABILIZAR PARCIALMENTE') | (output_df['BUCKET']=='4. RENTABILIZAR A TARGET')) & (output_df['VM_LM_NEW_X_PERC_TGMV'] - output_df['VM_lm']  < 0 )]
# output_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','NEW_PPM','BPC_NEW_X','VM_LM_NEW_X_PERC_TGMV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE']][((output_df['BUCKET']=='5. RENTABILIZAR PARCIALMENTE') | (output_df['BUCKET']=='4. RENTABILIZAR A TARGET')) & (output_df['FINAL_PPM_FLOOR'] - output_df['CURRENT_PPM_FLOOR']  < 0 )]

# #BPC
# output_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','NEW_PPM','BPC_NEW_X','VM_LM_NEW_X_PERC_TGMV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE']][((output_df['BUCKET']=='3. INVERTIR') | (output_df['BUCKET']=='2. COMPETITIVIZAR')) & (output_df['BPC_NEW_X'] - output_df['BPC_potencial']  < 0 )]
# output_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','NEW_PPM','BPC_NEW_X','VM_LM_NEW_X_PERC_TGMV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE']][((output_df['BUCKET']=='3. INVERTIR') | (output_df['BUCKET']=='2. COMPETITIVIZAR')) & (output_df['FINAL_PPM_FLOOR'] - output_df['CURRENT_PPM_FLOOR']  > 0 )]

##############################################################################
### SEGUNDO ROUND DA CALCULADORA #############################################
################################################################################

output_df_restricted = pd.DataFrame()
for i in range(0,len(self_representative_agg_brands)):
 
  example = self_representative_agg_brands.iloc[[i]]
  final_ppm_floor = output_df.iloc[i]['FINAL_PPM_FLOOR'].astype(int)
  current_ppm_floor = output_df.iloc[i]['CURRENT_PPM_FLOOR'].astype(int)

  if final_ppm_floor < current_ppm_floor - 5 :
    restricted_ppm_floor = current_ppm_floor - 5
  elif final_ppm_floor > current_ppm_floor + 5:
    restricted_ppm_floor = current_ppm_floor + 5
  else:
    restricted_ppm_floor = final_ppm_floor


  final_row, grid_df = bpc_calculator(df_bpc, df_agg_brands_inputs, example, min_ppm = restricted_ppm_floor, max_ppm = restricted_ppm_floor)
  output_df_restricted = pd.concat([output_df_restricted,final_row])
  print(i)


#Adicionar Flag de Top 20
output_df_restricted = output_df_restricted.merge(top20siteaggbrands, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND'), right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND'))
output_df_restricted['FLAG_top20_AGGBRAND'] = output_df_restricted['FLAG_top20_AGGBRAND'].fillna(0)


# Adding new columns to output_df_restricted
output_df_restricted['BUCKET']= output_df['BUCKET']

#Adicionar coluna de piso final
output_df_restricted['FINAL_PPM_FLOOR'] = np.nan
output_df_restricted['FINAL_PPM_FLOOR'][output_df_restricted['BUCKET'] == '1. MANTENER']= output_df_restricted['CURRENT_PPM_FLOOR']
output_df_restricted['FINAL_PPM_FLOOR'][output_df_restricted['BUCKET'] == '4. RENTABILIZAR A TARGET']= output_df_restricted[['NEW_PPM','CURRENT_PPM_FLOOR']].max(axis=1)
output_df_restricted['FINAL_PPM_FLOOR'][output_df_restricted['BUCKET'] == '5. RENTABILIZAR PARCIALMENTE']= output_df_restricted[['NEW_PPM','CURRENT_PPM_FLOOR']].max(axis=1)
output_df_restricted['FINAL_PPM_FLOOR'][output_df_restricted['BUCKET'] == '2. COMPETITIVIZAR']= output_df_restricted[['NEW_PPM','CURRENT_PPM_FLOOR']].min(axis=1)
output_df_restricted['FINAL_PPM_FLOOR'][output_df_restricted['BUCKET'] == '3. INVERTIR']= output_df_restricted[['NEW_PPM','CURRENT_PPM_FLOOR']].min(axis=1)
output_df_restricted['FINAL_PPM_FLOOR'][output_df_restricted['BUCKET'] == '6. REVISAR']= output_df_restricted['CURRENT_PPM_FLOOR']

output_df_restricted['BUCKET'][(output_df_restricted['FINAL_PPM_FLOOR']==output_df_restricted['CURRENT_PPM_FLOOR']) & (output_df_restricted['BUCKET']!='1. MANTENER')& (output_df_restricted['BUCKET']!='6. REVISAR')] = '1. MANTENER'


#Adicionar coluna de Governança
output_df_restricted['GOVERNANCE'] = 'ERROR'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR'].notna()) & (output_df_restricted['BUCKET'] == '1. MANTENER') ]= 'A. PRICING'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR'].notna()) & (output_df_restricted['BUCKET'] == '4. RENTABILIZAR A TARGET') & (output_df_restricted['FLAG_top20_AGGBRAND'] == 0)]= 'A. PRICING'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR'].notna()) & (output_df_restricted['BUCKET'] == '4. RENTABILIZAR A TARGET') & (output_df_restricted['FLAG_top20_AGGBRAND'] == 1)]= 'B. MANAGER/DIRECTOR'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR'].notna()) & (output_df_restricted['BUCKET'] == '5. RENTABILIZAR PARCIALMENTE')& (output_df_restricted['FLAG_top20_AGGBRAND'] == 0)]= 'A. PRICING'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR'].notna()) & (output_df_restricted['BUCKET'] == '5. RENTABILIZAR PARCIALMENTE')& (output_df_restricted['FLAG_top20_AGGBRAND'] == 1)]= 'B. MANAGER/DIRECTOR'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR'].notna()) & (output_df_restricted['BUCKET'] == '6. REVISAR')]= 'A. PRICING'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR']>=0) & (output_df_restricted['BUCKET'] == '2. COMPETITIVIZAR') & (output_df_restricted['FLAG_top20_AGGBRAND'] == 0)]= 'A. PRICING'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR']>=0) & (output_df_restricted['BUCKET'] == '2. COMPETITIVIZAR') & (output_df_restricted['FLAG_top20_AGGBRAND'] == 1) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_TGMV'] <= output_df['DC_PERC_L6M'] )]= 'B. MANAGER/DIRECTOR'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR']>=0) & (output_df_restricted['BUCKET'] == '3. INVERTIR') & (output_df_restricted['FLAG_top20_AGGBRAND'] == 0)]= 'A. PRICING'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR']>=0) & (output_df_restricted['BUCKET'] == '3. INVERTIR') & (output_df_restricted['FLAG_top20_AGGBRAND'] == 1) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_TGMV'] <= output_df['DC_PERC_L6M'] )]= 'B. MANAGER/DIRECTOR'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR']>=0) & (output_df_restricted['BUCKET'] == '2. COMPETITIVIZAR') & (output_df_restricted['FLAG_top20_AGGBRAND'] == 1) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_TGMV'] > output_df['DC_PERC_L6M'] )]= 'B. MANAGER/DIRECTOR'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR']>=0) & (output_df_restricted['BUCKET'] == '3. INVERTIR') & (output_df_restricted['FLAG_top20_AGGBRAND'] == 1) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_TGMV'] > output_df['DC_PERC_L6M'] )]= 'B. MANAGER/DIRECTOR'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR']<0) & (output_df_restricted['BUCKET'] == '2. COMPETITIVIZAR') ]= 'C. DIRECTOR/VP'
output_df_restricted['GOVERNANCE'][(output_df_restricted['FINAL_PPM_FLOOR']<0) & (output_df_restricted['BUCKET'] == '3. INVERTIR')]= 'C. DIRECTOR/VP'

pd.crosstab(output_df_restricted['BUCKET'][output_df_restricted['ITE_ATT_BRAND']!='ALL_BRANDS'],output_df_restricted['GOVERNANCE'],margins = True)



##########################################################
### CREATING EXECUTIVE SUMMARY                    ########
##########################################################

summary_df = output_df[['SIT_SITE_ID', 'VERTICAL', 'DOM_DOMAIN_AGG2', 'ITE_ATT_BRAND','BPC_original','BPC_ABC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','UE_CON_TGMV_AMT_LC_LM','TSI','TSI_NEW_X','BPC_NEW_X','UE_CON_TGMV_AMT_LC_LM_NEW_X','VM_LM_NEW_X_PERC_TGMV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE','DC_PERC_L6M','FLAG_top20_AGGBRAND']]

#Substituindo as colunas pós mudanças por sua versão restrita
summary_df['OPTIMAL_PPM_FLOOR']= summary_df['FINAL_PPM_FLOOR']
summary_df['FINAL_PPM_FLOOR'] = output_df_restricted['NEW_PPM']
summary_df['TSI_NEW_X'] = output_df_restricted['TSI_NEW_X']
summary_df['BPC_NEW_X'] = output_df_restricted['BPC_NEW_X']
summary_df['UE_CON_TGMV_AMT_LC_LM_NEW_X'] = output_df_restricted['UE_CON_TGMV_AMT_LC_LM_NEW_X']
summary_df['VM_LM_NEW_X_PERC_TGMV'] = output_df_restricted['VM_LM_NEW_X_PERC_TGMV']
summary_df['GOVERNANCE'] = output_df_restricted['GOVERNANCE']
summary_df['BUCKET'] = output_df_restricted['BUCKET']

#Trazendo os valores finais para os casos onde não mudamos piso
summary_df['FINAL_PPM_FLOOR'][(summary_df['BUCKET']=='1. MANTENER') | (summary_df['CURRENT_PPM_FLOOR']==summary_df['FINAL_PPM_FLOOR'])] = summary_df['CURRENT_PPM_FLOOR']
summary_df['BPC_NEW_X'][(summary_df['BUCKET']=='1. MANTENER') | (summary_df['CURRENT_PPM_FLOOR']==summary_df['FINAL_PPM_FLOOR'])] = summary_df['BPC_potencial']
summary_df['UE_CON_TGMV_AMT_LC_LM_NEW_X'][(summary_df['BUCKET']=='1. MANTENER') | (summary_df['CURRENT_PPM_FLOOR']==summary_df['FINAL_PPM_FLOOR'])] = summary_df['UE_CON_TGMV_AMT_LC_LM']
summary_df['VM_LM_NEW_X_PERC_TGMV'][(summary_df['BUCKET']=='1. MANTENER') | (summary_df['CURRENT_PPM_FLOOR']==summary_df['FINAL_PPM_FLOOR'])] = summary_df['VM_lm']
summary_df['TSI_NEW_X'][(summary_df['BUCKET']=='1. MANTENER') | (summary_df['CURRENT_PPM_FLOOR']==summary_df['FINAL_PPM_FLOOR'])] = summary_df['TSI']

#Criando colunas auxiliares para facilitar fazer análises post hoc
summary_df['VISITS_COMPETITIVE_POTENTIAL']= summary_df['BPC_potencial']*summary_df['VISITS_MATCH']
summary_df['VISITS_COMPETITIVE_POTENTIAL_NEW']= summary_df['BPC_NEW_X']*summary_df['VISITS_MATCH']
summary_df['UE_VM_LC']= summary_df['VM_lm']*summary_df['UE_CON_TGMV_AMT_LC_LM']
summary_df['UE_VM_LC_NEW']= summary_df['VM_LM_NEW_X_PERC_TGMV']*summary_df['UE_CON_TGMV_AMT_LC_LM_NEW_X']
summary_df['FLAG_ALL_BRANDS']= 0
summary_df['FLAG_ALL_BRANDS'][summary_df['ITE_ATT_BRAND']=='ALL_BRANDS']= 1

#Reordenando as colunas para facilitar a análise a posteriori
summary_df_rearranged = summary_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','CURRENT_PPM_FLOOR','OPTIMAL_PPM_FLOOR','FINAL_PPM_FLOOR','BPC_tgt','BPC_original','BPC_ABC_original','BPC_potencial','BPC_NEW_X','VM_tgt','VM_lm','VM_LM_NEW_X_PERC_TGMV','UE_CON_TGMV_AMT_LC_LM','UE_CON_TGMV_AMT_LC_LM_NEW_X','FLAG_top20_AGGBRAND','BUCKET','GOVERNANCE','VISITS_MATCH','VISITS_COMPETITIVE_POTENTIAL','VISITS_COMPETITIVE_POTENTIAL_NEW','UE_VM_LC','UE_VM_LC_NEW','UE_CON_TGMV_AMT_LC_LM','UE_CON_TGMV_AMT_LC_LM_NEW_X','TSI','TSI_NEW_X','FLAG_ALL_BRANDS','DC_PERC_L6M']]

#Renomeando colunas
summary_df_rearranged = summary_df_rearranged.rename(columns = {'BPC_NEW_X':'BPC_POTENCIAL_NEW','VM_tgt':'VM_tgt_OP','VM_lm':'VM_LAST_MONTH','VM_LM_NEW_X_PERC_TGMV':'VM_LAST_MONTH_NEW','DC_PERC_L6M':'DC_LAST_6_CLOSED_MONTHS','CURRENT_PPM_FLOOR':'AVERAGE_PPM_FLOOR_LAST_MONTH'})

#############################################
#Salvando o resultado

file_name = "summary_"+formatted_date+".xlsx"
summary_df_rearranged.to_excel(file_name,
             sheet_name='Sheet_name_1', index=False) 

# ##
# df_ts_08 = all_grids_df[(all_grids_df['ITE_ATT_BRAND']=='ALL_BRANDS') & (all_grids_df['BPC_NEW_X']< 0.8)][['SIT_SITE_ID','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_NEW_X','TSI_NEW_X','VM_LM_NEW_X_PERC_TGMV','UE_CON_TGMV_AMT_LC_LM_NEW_X']].groupby(['SIT_SITE_ID','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).first().reset_index()
# df_ts_06 = all_grids_df[(all_grids_df['ITE_ATT_BRAND']=='ALL_BRANDS') & (all_grids_df['BPC_NEW_X']< 0.6)][['SIT_SITE_ID','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_NEW_X','TSI_NEW_X','VM_LM_NEW_X_PERC_TGMV','UE_CON_TGMV_AMT_LC_LM_NEW_X']].groupby(['SIT_SITE_ID','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).first().reset_index()

# df_ts = df_ts_08.merge(df_ts_06, how = 'inner', left_on = ('SIT_SITE_ID','DOM_DOMAIN_AGG2','ITE_ATT_BRAND'), right_on = ('SIT_SITE_ID','DOM_DOMAIN_AGG2','ITE_ATT_BRAND'), suffixes = ('_08','_06'))

# df_ts.to_excel('output_ts_2025_bpc.xlsx',
#              sheet_name='Sheet_name_1', index=False) 
# ##