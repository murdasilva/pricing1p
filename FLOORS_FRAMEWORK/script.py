#OBS: RODAR EM VENV

import pandas as pd
pd.options.display.max_columns = 300
pd.options.display.max_rows = 70

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

##########################################################
### Finding the self representative AGG/Brands ###########
##########################################################

top50siteaggbrands= pd.DataFrame()
df_vm = df_agg_brands_inputs[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','UE_CON_TGMV_AMT_LC_L6CM','UE_CON_TGMV_AMT_LC_LM']].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()

for site in df_bpc['SIT_SITE_ID'].unique():
  df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC','VISITS_MATCH']][df_bpc['SIT_SITE_ID']==site].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
  currenttop50keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(50)
  currenttop50keys = currenttop50keys.merge(df_vm, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ),right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ) )
  currenttop50keys=currenttop50keys[currenttop50keys['VISITS_MATCH']>0]
  currenttop50keys = currenttop50keys[currenttop50keys['UE_CON_TGMV_AMT_LC_L6CM'] > 0]
  currenttop50keys = currenttop50keys[currenttop50keys['UE_CON_TGMV_AMT_LC_LM'] > 0]
  top50siteaggbrands = pd.concat([top50siteaggbrands,currenttop50keys])


top10verticalsaggbrands= pd.DataFrame()

for site in df_bpc['SIT_SITE_ID'].unique():
    for vertical in df_bpc['VERTICAL'].unique():
        df_tgmv = df_bpc[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','TGMV_LC','VISITS_MATCH']][(df_bpc['SIT_SITE_ID']==site) & (df_bpc['VERTICAL']==vertical)].groupby(['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']).sum().reset_index()
        currenttop10keys = df_tgmv.sort_values('TGMV_LC', ascending = False).head(10)
        currenttop10keys = currenttop10keys.merge(df_vm, how = 'left', left_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ),right_on = ('SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND' ) )
        currenttop10keys=currenttop10keys[currenttop10keys['VISITS_MATCH']>0]
        currenttop10keys = currenttop10keys[currenttop10keys['UE_CON_TGMV_AMT_LC_L6CM'] > 0]
        currenttop10keys = currenttop10keys[currenttop10keys['UE_CON_TGMV_AMT_LC_LM'] > 0]
        top10verticalsaggbrands = pd.concat([top10verticalsaggbrands,currenttop10keys])



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
      top50siteaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
      ,top10verticalsaggbrands[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
      ,aggkeys[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND']]
    ]).drop_duplicates().sort_values(by = ['SIT_SITE_ID','VERTICAL']).reset_index(drop=True)


##########################################################
### STARTING A PROOF OF CONCEPT OF THE CALCULATOR ########
##########################################################

output_df = pd.DataFrame()


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

for i in range(0,len(self_representative_agg_brands)):
 
  example = self_representative_agg_brands.iloc[[i]]

          
  mask_bpc = mask_function(df_bpc,example)
  mask_inputs = mask_function(df_agg_brands_inputs,example)



  df_bpc_filtered = df_bpc[ mask_bpc ]


  df_bpc_filtered['VISITS_COMPETITIVE_POTENTIAL']=np.where(df_bpc_filtered[['PPM_CALCULATED_FLOOR_PRICE','PRICE_MELI2']].min(axis=1)<=1.01*df_bpc_filtered['COMP_PRICE_RIVAL'],df_bpc_filtered['VISITS_MATCH'],0)

  df_agg_brands_inputs_filtered = df_agg_brands_inputs[ mask_inputs ]



  newdf = example.copy()

  newdf['BPC_original']= df_bpc_filtered['VISITS_COMPETITIVE'].sum()/df_bpc_filtered['VISITS_MATCH'].sum()
  newdf['BPC_estimado']= df_bpc_filtered['VISITS_COMPETITIVE_ESTIMATED'].sum()/df_bpc_filtered['VISITS_MATCH'].sum()
  newdf['BPC_potencial']= df_bpc_filtered['VISITS_COMPETITIVE_POTENTIAL'].sum()/df_bpc_filtered['VISITS_MATCH'].sum()
  # newdf['BPC_NEW_0']= sum(df_bpc_filtered['VISITS_COMPETITIVE_NEW_0'])/sum(df_bpc_filtered['VISITS_MATCH'])

  newdf['BPC_tgt']= df_agg_brands_inputs_filtered['TARGET_PRIORIZED'].mean()
  newdf['VISITS_MATCH'] = df_bpc_filtered['VISITS_MATCH'].sum()

  newdf['VM_lm']= df_agg_brands_inputs_filtered['UE_MNG_VENDOR_MARGIN_AMT_LC_LM'].sum()/df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_LM'].sum()
  newdf['VM_tgt']= df_agg_brands_inputs_filtered['TGT_VENDOR_MARGIN_PERC_REV'].mean()

  newdf['UE_MNG_REVENUE_GROSS_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_LM'].sum()
  newdf['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'].sum()
  newdf['UE_CON_CMV_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_CON_CMV_AMT_LC_LM'].sum()
  newdf['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM'].sum()
  newdf['UE_CON_CONTRACOGS_AMT_LC_LM'] = df_agg_brands_inputs_filtered['UE_CON_CONTRACOGS_AMT_LC_LM'].sum()




  newdf['VM_l6cm']= df_agg_brands_inputs_filtered['UE_MNG_VENDOR_MARGIN_AMT_LC_L6CM'].sum()/df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'].sum()
  newdf['VC_l6cm']= df_agg_brands_inputs_filtered['UE_MNG_VARIABLE_CONTRIBUTION_AMT_LC_L6CM'].sum()/df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'].sum()
  newdf['DC_l6cm']= df_agg_brands_inputs_filtered['UE_MNG_DIRECT_CONTRIBUTION_AMT_LC_L6CM'].sum()/df_agg_brands_inputs_filtered['UE_MNG_REVENUE_GROSS_AMT_LC_L6CM'].sum()

  newdf['TGMV_LC']= df_bpc_filtered['TGMV_LC'].sum()
  newdf['TGMV_LC_ESTIMATED']= df_bpc_filtered['TGMV_LC_ESTIMATED'].sum()

  newdf['TSI'] = df_bpc_filtered['TSI'].sum()
  # newdf['TGMV_LC_NEW_0']= sum(df_bpc_filtered['TGMV_LC_NEW_0'])

  grid_df = pd.DataFrame()

  for new_ppm in range(-10, 55 + 1):

    df_bpc_filtered['PRICE_TO_CHASE_X'] = np.where(df_bpc_filtered['COMP_PRICE_RIVAL'].isna(),df_bpc_filtered['PRICE_MELI2'], df_bpc_filtered[['COMP_PRICE_RIVAL','PRICE_MELI2']].values.min(1))
    df_bpc_filtered['PPM_CALCULATED_FLOOR_PRICE_X'] = ((df_bpc['COST']-df_bpc['CCOGS'])*df_bpc['SIT_SITE_IVA'])/(1-new_ppm/100-df_bpc['FINANCIAL_COST']/100)
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


    new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM'] * new_row['TGMV_LC_NEW_X']/new_row['TGMV_LC_ESTIMATED']
    new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM'] * 1
    new_row['UE_CON_CMV_AMT_LC_LM_NEW_X'] =  new_row['UE_CON_CMV_AMT_LC_LM'] * new_row['TSI_NEW_X']/new_row['TSI']
    new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM_NEW_X'] = new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM']* new_row['TSI_NEW_X']/new_row['TSI']
    new_row['UE_CON_CONTRACOGS_AMT_LC_LM_NEW_X'] = new_row['UE_CON_CONTRACOGS_AMT_LC_LM'] * 1
    
    new_row['VM_LM_NEW_X_PERC_REV'] = (new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X'] + new_row['UE_MNG_NON_BANK_COUPONS_DISCOUNT_AMT_LC_LM_NEW_X'] + new_row['UE_CON_CMV_AMT_LC_LM_NEW_X'] + new_row['UE_MNG_OTHER_PRODUCT_COST_AMT_LC_LM_NEW_X'] + new_row['UE_CON_CONTRACOGS_AMT_LC_LM_NEW_X'])/(new_row['UE_MNG_REVENUE_GROSS_AMT_LC_LM_NEW_X']) 
    
    new_row['CURRENT_PPM_FLOOR'] = round((df_bpc_filtered[df_bpc_filtered['PPM_PROFIT_FLOOR'].notna()]['PPM_PROFIT_FLOOR']*df_bpc_filtered[df_bpc_filtered['PPM_PROFIT_FLOOR'].notna()]['VISITS_MATCH']).sum()/df_bpc_filtered[df_bpc_filtered['PPM_PROFIT_FLOOR'].notna()]['VISITS_MATCH'].sum(),0)
    

    grid_df = pd.concat([grid_df,new_row])
  
  ppm_that_yields_both_tgts = grid_df[(grid_df['BPC_NEW_X'] >= grid_df['BPC_tgt']) & (grid_df['VM_LM_NEW_X_PERC_REV'] >= grid_df['VM_tgt']) ].tail(5).head(1) # PPM that allows for both targets  (.tail(5) garante que nesses casos ficamos perto do VM_target)
  ppm_that_yields_bpc_tgt = grid_df[grid_df['BPC_NEW_X'] >= grid_df['BPC_tgt']].tail(1) # Greatest PPM that allows for BPC tgt
  ppm_min = grid_df.head(1) #Smallest allowed PPM 

  final_df = pd.concat([ppm_that_yields_both_tgts,ppm_that_yields_bpc_tgt,ppm_min])
  final_row = final_df.iloc[[0]]

  output_df = pd.concat([output_df,final_row])

  print(i)


# Adding new columns to output_df

output_df['BUCKET'] = np.nan
output_df['BUCKET'][(output_df['VM_lm']>=output_df['VM_tgt']) & (output_df[['BPC_original','BPC_potencial']].max(axis=1)>=output_df['BPC_tgt'])  ] = '1. MANTENER'
output_df['BUCKET'][(output_df['VM_lm']<output_df['VM_tgt'])  & (output_df[['BPC_original','BPC_potencial']].max(axis=1)>=output_df['BPC_tgt']) & (output_df['BPC_NEW_X']>=output_df['BPC_tgt']) & (output_df['VM_LM_NEW_X_PERC_REV']>=output_df['VM_tgt'])] = '4. RENTABILIZAR A TARGET'
output_df['BUCKET'][(output_df['VM_lm']<output_df['VM_tgt'])  & (output_df[['BPC_original','BPC_potencial']].max(axis=1)>=output_df['BPC_tgt']) & (output_df['BPC_NEW_X']>=output_df['BPC_tgt']) & (output_df['VM_LM_NEW_X_PERC_REV']<output_df['VM_tgt'])] = '5. RENTABILIZAR PARCIALMENTE'
output_df['BUCKET'][(output_df[['BPC_original','BPC_potencial']].max(axis=1)<output_df['BPC_tgt']) & (output_df['VM_LM_NEW_X_PERC_REV']>=output_df['VM_tgt']) & (output_df['BPC_NEW_X']>=output_df['BPC_tgt']) ] = '2. COMPETITIVIZAR'
output_df['BUCKET'][(output_df[['BPC_original','BPC_potencial']].max(axis=1)<output_df['BPC_tgt']) & (output_df['VM_LM_NEW_X_PERC_REV']<output_df['VM_tgt']) & (output_df['BPC_NEW_X']>=output_df['BPC_tgt']) ] = '3. INVERTIR'
output_df['BUCKET'][(output_df['BPC_NEW_X']<output_df['BPC_tgt']) ] = '6. REVISAR'

#Adicionar coluna de piso final
output_df['FINAL_PPM_FLOOR'] = np.nan
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '1. MANTENER']= output_df['CURRENT_PPM_FLOOR']
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '4. RENTABILIZAR A TARGET']= output_df[['NEW_PPM','CURRENT_PPM_FLOOR']].max(axis=1)
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '5. RENTABILIZAR PARCIALMENTE']= output_df[['NEW_PPM','CURRENT_PPM_FLOOR']].max(axis=1)
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '2. COMPETITIVIZAR']= output_df[['NEW_PPM','CURRENT_PPM_FLOOR']].min(axis=1)
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '3. INVERTIR']= output_df[['NEW_PPM','CURRENT_PPM_FLOOR']].min(axis=1)
output_df['FINAL_PPM_FLOOR'][output_df['BUCKET'] == '6. REVISAR']= output_df['CURRENT_PPM_FLOOR']

#Adicionar coluna de Governança

output_df['GOVERNANCE'] = np.nan
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'] < 0) & ((output_df['BUCKET'] == '2. COMPETITIVIZAR') |(output_df['BUCKET'] == '3. INVERTIR')  ) ]= 'C. DIRECTOR/VP'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '1. MANTENER')]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'] >= 0) & (output_df['BUCKET'] == '2. COMPETITIVIZAR')]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '4. RENTABILIZAR A TARGET')]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'].notna()) & (output_df['BUCKET'] == '5. RENTABILIZAR PARCIALMENTE')]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'] >= 0) & (output_df['BUCKET'] == '3. INVERTIR') & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_REV'] <= 0.01 ) ]= 'A. PRICING'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'] >= 0) & (output_df['BUCKET'] == '3. INVERTIR') & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_REV'] > 0.01 ) ]= 'B. MANAGER/DIRECTOR'
# output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'] >= 0) & (output_df['BUCKET'] == '3. INVERTIR') & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_REV'] > 0.01 ) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_REV'] <= 0.05 )]= 'B. MANAGER/DIRECTOR'
# output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'] >= 0) & (output_df['BUCKET'] == '3. INVERTIR') & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_REV'] > 0.01 ) & (output_df['VM_lm'] - output_df['VM_LM_NEW_X_PERC_REV'] > 0.05 )]= 'C. DIRECTOR/VP'
output_df['GOVERNANCE'][(output_df['FINAL_PPM_FLOOR'] .notna()) & (output_df['BUCKET'] == '6. REVISAR')]= 'A. PRICING'
tsi_threshold = 1000 # Não levar a VP AGGs de baixa importância em TSI
#output_df['GOVERNANCE'][ ((output_df['GOVERNANCE']== 'CB. MANAGER/DIRECTOR') | (output_df['GOVERNANCE']== 'C. DIRECTOR/VP')) & (output_df['TSI'] < tsi_threshold)] = 'B. MANAGER'

output_df.describe()

pd.crosstab(output_df['BUCKET'][output_df['ITE_ATT_BRAND']!='ALL_BRANDS'],output_df['GOVERNANCE'],margins = True)
pd.crosstab(output_df['BUCKET'][output_df['ITE_ATT_BRAND']=='ALL_BRANDS'],output_df['GOVERNANCE'],margins = True)

# Visitas por top AGG2/BRANDS
pd.pivot_table(output_df[output_df['ITE_ATT_BRAND']!='ALL_BRANDS'], values=['VISITS_MATCH'], index=['BUCKET'], columns=['GOVERNANCE'], aggfunc=np.sum, margins=True,fill_value = 0)
# Visitas por AGG2
pd.pivot_table(output_df[output_df['ITE_ATT_BRAND']=='ALL_BRANDS'], values=['VISITS_MATCH'], index=['BUCKET'], columns=['GOVERNANCE'], aggfunc=np.sum, margins=True,fill_value = 0)


# CHECAR SE TEM CASOS ASSIM
output_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','NEW_PPM','BPC_NEW_X','VM_LM_NEW_X_PERC_REV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE']][((output_df['BUCKET']=='5. RENTABILIZAR PARCIALMENTE') | (output_df['BUCKET']=='4. RENTABILIZAR A TARGET')) & (output_df['VM_LM_NEW_X_PERC_REV'] - output_df['VM_lm']  < 0 )]
output_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','NEW_PPM','BPC_NEW_X','VM_LM_NEW_X_PERC_REV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE']][((output_df['BUCKET']=='5. RENTABILIZAR PARCIALMENTE') | (output_df['BUCKET']=='4. RENTABILIZAR A TARGET')) & (output_df['FINAL_PPM_FLOOR'] - output_df['CURRENT_PPM_FLOOR']  < 0 )]

#BPC
output_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','NEW_PPM','BPC_NEW_X','VM_LM_NEW_X_PERC_REV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE']][((output_df['BUCKET']=='3. INVERTIR') | (output_df['BUCKET']=='2. COMPETITIVIZAR')) & (output_df['BPC_NEW_X'] - output_df['BPC_potencial']  < 0 )]
output_df[['SIT_SITE_ID','VERTICAL','DOM_DOMAIN_AGG2','ITE_ATT_BRAND','BPC_original','BPC_potencial','BPC_tgt','VISITS_MATCH','VM_lm','VM_tgt','NEW_PPM','BPC_NEW_X','VM_LM_NEW_X_PERC_REV','CURRENT_PPM_FLOOR','FINAL_PPM_FLOOR','BUCKET','GOVERNANCE']][((output_df['BUCKET']=='3. INVERTIR') | (output_df['BUCKET']=='2. COMPETITIVIZAR')) & (output_df['FINAL_PPM_FLOOR'] - output_df['CURRENT_PPM_FLOOR']  > 0 )]

#############################################

output_df.to_excel("output.xlsx",
             sheet_name='Sheet_name_1') 
