#OBS: RODAR EM VENV

import pandas as pd
import requests as req
import numpy as np
import datetime
from google.cloud import bigquery
client = bigquery.Client()




sql = f"""
SELECT PO.SIT_SITE_ID
  ,PO.ITE_ITEM_ID
  ,R.ITEM_ID AS ITEM_ID_HERMANADO
  ,COALESCE(R.ITEM_ID , PO.ITE_ITEM_ID) AS ITE_ITEM_ID_REF
    ,CONCAT(PO.SIT_SITE_ID, CAST(PO.ITE_ITEM_ID AS STRING)) AS MLB_ORIG
  ,CONCAT(PO.SIT_SITE_ID, CAST(COALESCE(R.ITEM_ID , PO.ITE_ITEM_ID) AS STRING) ) AS MLB_REF
 FROM `meli-bi-data.WHOWNER.LK_PL1P_PRICING_OPPS` PO
 LEFT JOIN WHOWNER.LK_ITE_ITEM_DOMAINS DOM ON PO.SIT_SITE_ID = DOM.SIT_SITE_ID AND PO.ITE_ITEM_ID = DOM.ITE_ITEM_ID
 LEFT JOIN WHOWNER.LK_ITE_ITEMS ITE ON PO.SIT_SITE_ID = ITE.SIT_SITE_ID AND PO.ITE_ITEM_ID = ITE.ITE_ITEM_ID, unnest (ITE_ITEM_RELATIONS) as R
 WHERE PO.SIT_SITE_ID = 'MLB'

AND PO.ITE_ITEM_STATUS = 'ACTIVE'
AND UPPER(ITE_ATT_BRAND) = 'CETAPHIL'
AND DOM.DOM_DOMAIN_AGG2 = 'PERSONAL CARE'
AND LAST_STATUS_FLG = 1
AND ITE.ITE_ITEM_CATALOG_LISTING_FLG = TRUE
ORDER BY ITE_ITEM_ID 

"""

df = client.query_and_wait(sql).to_dataframe()




mydf = pd.DataFrame()

for i in range(len(df)):
    api_pricing_response = req.get(f'https://internal-api.mercadolibre.com/1p-pricing-api/item/pricing/{df['MLB_REF'][i]}')
    ptw_response =  req.get(f'https://internal-api.mercadolibre.com/items/{df['MLB_ORIG'][i]}/buy_box/price_to_win?version=2')
    
    
    price = float(api_pricing_response.json()['result']['price'])
    strategy = str(api_pricing_response.json()['result']['strategy_type'])
    buybox_status = str(ptw_response.json()['status'])
    visit_share = str(ptw_response.json()['visit_share'])

    newdf = pd.DataFrame( 
        {'item':[df['MLB_ORIG'][i]],
        'price':[price],
        'strategy':[strategy],
        'buybox_status':[buybox_status],
        'visit_share':[visit_share]
        }
    ).set_index('item')

    mydf = pd.concat([mydf,newdf])


mydf.to_excel(f'Catalog_items_photo_{str(datetime.datetime.now()).replace('-','_').replace(' ', '_').replace(':','_').replace('.','_')}.xlsx')

mydf

