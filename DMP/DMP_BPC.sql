SELECT
  FAV.TIM_DAY,
        EXTRACT(YEAR FROM DATE_TRUNC(FAV.TIM_DAY, WEEK)) *100 + EXTRACT(WEEK FROM DATE_TRUNC(FAV.TIM_DAY, WEEK)) AS WEEK,
    EXTRACT(YEAR FROM FAV.TIM_DAY) *100 + EXTRACT(MONTH FROM FAV.TIM_DAY) AS MONTH,
    EXTRACT(YEAR FROM FAV.TIM_DAY) *100 + EXTRACT(QUARTER FROM FAV.TIM_DAY) AS QUARTER,
    CASE WHEN EXTRACT (YEAR FROM DATE_TRUNC(FAV.TIM_DAY, WEEK)) = EXTRACT(YEAR FROM DATE_ADD(DATE_TRUNC(FAV.TIM_DAY, WEEK), INTERVAL 6 DAY)) 
      THEN
        CONCAT(CAST(DATE_TRUNC(FAV.TIM_DAY, WEEK)  AS STRING), " to ",CAST(DATE_ADD(DATE_TRUNC(FAV.TIM_DAY, WEEK), INTERVAL 6 DAY) AS STRING))
      ELSE 
            CONCAT(CAST(DATE_TRUNC(FAV.TIM_DAY, WEEK)  AS STRING), " to ",CAST(DATE_ADD(DATE_TRUNC(FAV.TIM_DAY, WEEK), INTERVAL 6 DAY) AS STRING))
            --CONCAT(CAST(DATE_TRUNC(P.TIM_DAY, WEEK)  AS STRING), " to ",CAST(DATE(EXTRACT (YEAR FROM DATE_TRUNC(P.TIM_DAY, WEEK)),12,31) AS STRING)) 
      END AS WEEK_DETAIL,
  FAV.SIT_SITE_ID,
  FAV.ITE_ITEM_ID,
  FAV.VERTICAL,
  CASE 
        WHEN FAV.DOM_AGG_1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL') THEN 'APPAREL'
        WHEN FAV.DOM_AGG_1 IN ('BEAUTY') THEN 'BEAUTY'
        ELSE NULL END AS SUBVERTICAL,
  FAV.DOM_AGG_1 AS DOM_DOMAIN_AGG1,
  FAV.DOM_AGG_2 AS DOM_DOMAIN_AGG2,
  FAV.BRAND AS ITE_ATT_BRAND,
  VD.SAP_VENDOR_ESTIMATED,
  FAV.FAVORABILITY_TYPE,
  LKI.ITE_ITEM_SUPERMARKET_FLG,
  LKI.ITE_ITEM_SCHEDULED_FLG,
  SUM(FAV.VISITS_EXPENSIVE) AS VISITS_EXPENSIVE,
  SUM(FAV.VISITS_MATCH) AS VISITS_MATCH,
FROM `WHOWNER.BT_COM_FAVORABILITY` FAV
LEFT JOIN     
  (SELECT *
    , ROW_NUMBER() OVER(PARTITION BY SIT_SITE_ID, ITE_ITEM_ID ORDER BY CASE WHEN SAP_VENDOR_ESTIMATED  IS NOT NULL THEN 1 ELSE 0 END DESC ) AS RW 
    FROM meli-bi-data.SBOX_PLANNING_1P.VW_BRANDS_VENDORS_ITEM_ID 
    QUALIFY RW =1
    ) VD 
 ON VD.SIT_SITE_ID = FAV.SIT_SITE_ID AND VD.ITE_ITEM_ID = FAV.ITE_ITEM_ID
LEFT JOIN WHOWNER.LK_ITE_ITEMS LKI ON FAV.SIT_SITE_ID = LKI.SIT_SITE_ID AND FAV.ITE_ITEM_ID = LKI.ITE_ITEM_ID
--LEFT JOIN WHOWNER.LK_ITE_ITEM_DOMAINS DOM ON FAV.SIT_SITE_ID = DOM.SIT_SITE_ID AND FAV.ITE_ITEM_ID = DOM.ITE_ITEM_ID
WHERE FAV.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO')
  AND FAV.FAVORABILITY_TYPE IN ('BOX_1P_AB' , 'BOX_1P_ABC' ,  'LANDED_1P_AB' ,  'LANDED_1P_ABC','BOX_1P_A','BOX_1P_AB_PIX','BOX_1P_ABC_PIX','LANDED_1P_A','1P_AB_CUOTAS','1P_ABC_CUOTAS')
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16