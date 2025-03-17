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
