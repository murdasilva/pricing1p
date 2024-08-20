-- version atualizada en 2024-03-21 por murilo.dasilva@mercadolivre.com

SELECT
    P.TIM_DAY,
    CAST(CASE WHEN EXTRACT(QUARTER FROM P.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),'0',EXTRACT(QUARTER FROM P.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),EXTRACT(QUARTER FROM P.TIM_DAY)) END AS INT) AS QUARTER,  
    CAST(CASE WHEN EXTRACT(MONTH FROM P.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),'0',EXTRACT(MONTH FROM P.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),EXTRACT(MONTH FROM P.TIM_DAY)) END AS INT) AS MONTH,  
    CAST(CASE WHEN EXTRACT(WEEK FROM P.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),'0',EXTRACT(WEEK FROM P.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),EXTRACT(WEEK FROM P.TIM_DAY)) END AS INT) AS WEEK,  
    CASE WHEN EXTRACT (YEAR FROM DATE_TRUNC(P.TIM_DAY, WEEK)) = EXTRACT(YEAR FROM DATE_ADD(DATE_TRUNC(P.TIM_DAY, WEEK), INTERVAL 6 DAY)) 
      THEN
        CONCAT(CAST(DATE_TRUNC(P.TIM_DAY, WEEK)  AS STRING), " to ",CAST(DATE_ADD(DATE_TRUNC(P.TIM_DAY, WEEK), INTERVAL 6 DAY) AS STRING))
      ELSE 
        CONCAT(CAST(DATE_TRUNC(P.TIM_DAY, WEEK)  AS STRING), " to ",CAST(DATE(EXTRACT (YEAR FROM DATE_TRUNC(P.TIM_DAY, WEEK)),12,31) AS STRING)) 
      END AS WEEK_DETAIL,
    P.SIT_SITE_ID,
    P.ITE_ITEM_ID,
    P.VERTICAL,
    CASE 
        WHEN P.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL') THEN 'APPAREL'
        WHEN P.DOM_DOMAIN_AGG1 IN ('BEAUTY') THEN 'BEAUTY'
        ELSE NULL END AS SUBVERTICAL,
    P.DOM_DOMAIN_AGG1,
    P.DOM_DOMAIN_AGG2,
    P.ITE_ATT_BRAND,
    VD.SAP_VENDOR_ESTIMATED,
    LKI.ITE_ITEM_SUPERMARKET_FLG,
    LKI.ITE_ITEM_SCHEDULED_FLG,  
    SUM(CASE WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY IN ('PROFITABILITY_PRICE') then  P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES ELSE 0 END) AS VISITS_PROFIT_FLOOR,
    SUM(P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES) AS  VISITS_TOTAL  
  FROM meli-bi-data.WHOWNER.DM_PL1P_PRICING_PERFORMANCE as P
  LEFT JOIN `meli-bi-data.SBOX_PRICING1P.VISITAS_TOTALES` AS V ON P.TIM_DAY = V.TIM_DAY AND P.SIT_SITE_ID = V.SIT_SITE_ID AND P.ITE_ITEM_ID = V.ITE_ITEM_ID
  LEFT JOIN     
    (SELECT *
      , ROW_NUMBER() OVER(PARTITION BY SIT_SITE_ID, ITE_ITEM_ID ORDER BY CASE WHEN SAP_VENDOR_ESTIMATED  IS NOT NULL THEN 1 ELSE 0 END DESC ) AS RW 
      FROM meli-bi-data.SBOX_PLANNING_1P.VW_BRANDS_VENDORS_ITEM_ID 
      QUALIFY RW =1
      ) VD ON P.SIT_SITE_ID = VD.SIT_SITE_ID AND P.ITE_ITEM_ID = VD.ITE_ITEM_ID
  LEFT JOIN WHOWNER.LK_ITE_ITEMS LKI ON P.SIT_SITE_ID = LKI.SIT_SITE_ID AND P.ITE_ITEM_ID = LKI.ITE_ITEM_ID
  WHERE P.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO')
    AND P.TIM_DAY > '2022-12-01'
    AND P.ITE_ITEM_STATUS = 'ACTIVE'
    --AND P.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL','BEAUTY')
  GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
