WITH AP_SIMPLIFIED AS (

SELECT DA.TIM_DAY
  ,DA.SIT_SITE_ID
  ,DA.ITE_ITEM_ID
  ,DA.VERTICAL
  ,DA.DOM_DOMAIN_AGG1
  ,DA.DOM_DOMAIN_AGG2
  ,DA.ITE_ATT_BRAND
  ,SUM(DA.VISITS_AUTOMATICL_PRICE) AS VISITS_AUTOMATICL_PRICE
  ,SUM(DA.VISITS_TOTAL) AS VISITS_TOTAL
FROM `SBOX_PRICING1P.DMP_AP` DA
WHERE DA.TIM_DAY >= '2024-01-01'
GROUP BY ALL
)

,BRANDS_1P AS (
  SELECT SIT_SITE_ID
    ,ITE_ITEM_ID
    ,K.VALUE_NAME as ITE_ATT_BRAND_ITE_ITEMS
    --, ROW_NUMBER() OVER (PARTITION BY SIT_SITE_ID, ITE_ITEM_ID ) AS RW
  FROM WHOWNER.LK_ITE_ITEMS, UNNEST(ITE_ITEM_ATTRIBUTES) K 
  WHERE K.ID = 'BRAND'
    AND ITE_ITEM_PARTY_TYPE_ID = '1P'
  --QUALIFY RW = 1


)



SELECT ASI.*
  ,DOM.VERTICAL AS VERTICAL_ITE_ITEM_DOMAINS
  ,DOM.DOM_DOMAIN_AGG1 AS DOM_DOMAIN_AGG1_ITE_ITEM_DOMAINS
  ,DOM.DOM_DOMAIN_AGG2 AS DOM_DOMAIN_AGG2_ITE_ITEM_DOMAINS
  ,IT.ITE_ATT_BRAND_ITE_ITEMS
FROM AP_SIMPLIFIED AS ASI
LEFT JOIN `meli-bi-data.WHOWNER.LK_ITE_ITEM_DOMAINS` DOM ON ASI.SIT_SITE_ID = DOM.SIT_SITE_ID AND ASI.ITE_ITEM_ID = DOM.ITE_ITEM_ID
LEFT JOIN BRANDS_1P IT  ON ASI.SIT_SITE_ID = IT.SIT_SITE_ID AND ASI.ITE_ITEM_ID = IT.ITE_ITEM_ID 
