#OBS: RODAR EM VENV

import pandas as pd
import requests as req
import numpy as np

from google.cloud import bigquery
client = bigquery.Client()



##########################################################
### DEFININDO O ITEM ID A SER INVESTIGADO ################
##########################################################


def import_sql(file_path):
  with open(file_path, 'r', encoding = 'utf-8') as file:
    sql_script = file.read()

    return sql_script

oi = import_sql('query_BPC.sql')


df = client.query_and_wait(oi).to_dataframe()