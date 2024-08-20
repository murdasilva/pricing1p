-- PC --
-- version atualizada en 2024-03-22 por murilo.dasilva@mercadolivre.com

SELECT 
  P.TIM_DAY,
    CAST(CASE WHEN EXTRACT(WEEK FROM P.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),'0',EXTRACT(WEEK FROM P.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),EXTRACT(WEEK FROM P.TIM_DAY)) END AS INT) AS WEEK,  
    --CASE WHEN EXTRACT(WEEK FROM P.TIM_DAY) < 10 THEN CONCAT(P.SIT_SITE_ID,P.VERTICAL,EXTRACT(YEAR FROM P.TIM_DAY),'0',EXTRACT(WEEK FROM P.TIM_DAY)) ELSE CONCAT(P.SIT_SITE_ID,P.VERTICAL,EXTRACT(YEAR FROM P.TIM_DAY),EXTRACT(WEEK FROM P.TIM_DAY)) END AS KEY_WEEK,  
    CAST(CASE WHEN EXTRACT(MONTH FROM P.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),'0',EXTRACT(MONTH FROM P.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),EXTRACT(MONTH FROM P.TIM_DAY)) END AS INT) AS MONTH,  
    --CASE WHEN EXTRACT(MONTH FROM P.TIM_DAY) < 10 THEN CONCAT(P.SIT_SITE_ID,P.VERTICAL,EXTRACT(YEAR FROM P.TIM_DAY),'0',EXTRACT(MONTH FROM P.TIM_DAY)) ELSE CONCAT(P.SIT_SITE_ID,P.VERTICAL,EXTRACT(YEAR FROM P.TIM_DAY),EXTRACT(MONTH FROM P.TIM_DAY)) END AS KEY_MONTH,  
    CAST(CASE WHEN EXTRACT(QUARTER FROM P.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),'0',EXTRACT(QUARTER FROM P.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM P.TIM_DAY),EXTRACT(QUARTER FROM P.TIM_DAY)) END AS INT) AS QUARTER,  
    CASE WHEN EXTRACT (YEAR FROM DATE_TRUNC(P.TIM_DAY, WEEK)) = EXTRACT(YEAR FROM DATE_ADD(DATE_TRUNC(P.TIM_DAY, WEEK), INTERVAL 6 DAY)) 
      THEN
        CONCAT(CAST(DATE_TRUNC(P.TIM_DAY, WEEK)  AS STRING), " to ",CAST(DATE_ADD(DATE_TRUNC(P.TIM_DAY, WEEK), INTERVAL 6 DAY) AS STRING))
      ELSE 
        CONCAT(CAST(DATE_TRUNC(P.TIM_DAY, WEEK)  AS STRING), " to ",CAST(DATE(EXTRACT (YEAR FROM DATE_TRUNC(P.TIM_DAY, WEEK)),12,31) AS STRING)) 
      END AS WEEK_DETAIL,
    P.SIT_SITE_ID,
    P.ITE_ITEM_ID,
    --SAFE_CAST(SK.ITE_ITEM_SAP_SKU AS BIGINT) as SAP_SKU,
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
    SUM(CASE WHEN P.PI_VISTA < 1.01 THEN P.VISITS ELSE PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES END) AS ERROR_COMPETITIVENESS,
    
    SUM(CASE WHEN P.PI_VISTA BETWEEN 0 AND 1.01 OR PL1P_PRICING_CURRENT_WINNING_STRATEGY IN ('MATCH_EXT','ADJUSTED_PRICE') -- 4/8/2023 saco 'COMPETITORS_SCORE_A_PRICE','ADJUSTED_SCORE_A_PRICE'
        THEN P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES ELSE 0 END) AS BOX_COMPETITIVENESS,
    SUM(CASE WHEN SAFE_DIVIDE(ITE_SITE_CURRENT_PRICE,COMP_ITEM_PRICE_ADJUSTED) < 1.01 OR PL1P_PRICING_CURRENT_WINNING_STRATEGY IN ('MATCH_EXT','ADJUSTED_PRICE','COMPETITORS_SCORE_A_PRICE','ADJUSTED_SCORE_A_PRICE') 
        THEN P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES ELSE 0 END) AS LANDED_COMPETITIVENESS,
    /*
    SUM(CASE WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY IN ('MATCH_EXT','ADJUSTED_PRICE') THEN P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES -- 4/8/2023 saco 'COMPETITORS_SCORE_A_PRICE','ADJUSTED_SCORE_A_PRICE'
             WHEN P.SIT_SITE_ID = 'MLM' AND ITE_ATT_BRAND IN ('SAMSUNG','HUAWEI') AND SAFE_DIVIDE(ITE_SITE_CURRENT_PRICE,COMP_ITEM_PRICE_ADJUSTED) BETWEEN 0 AND 1.01 THEN P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES 
             WHEN P.SIT_SITE_ID = 'MLM' AND CUS_CUST_ID = 516445073 AND SAFE_DIVIDE(ITE_SITE_CURRENT_PRICE,COMP_ITEM_PRICE_ADJUSTED) BETWEEN 0 AND 1.01 THEN P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES 
             WHEN P.SIT_SITE_ID = 'MLB' AND SAFE_DIVIDE(ITE_SITE_CURRENT_PRICE,GREATEST(COMP_ITEM_PRICE,COMP_ITEM_PRICE_ADJUSTED)) BETWEEN 0 AND 1.01 
                  AND (CUS_CUST_ID IN (480263032,480265022,418149407) AND DOM_DOMAIN_AGG2 IN ('TELEVISIONS','REFRIGERATION','LAUNDRY & DISHWASHERS','BUILT IN','PERSONAL CARE','MANICURE & PEDICURE') 
                    OR CUS_CUST_ID IN (480263032,451403353,768826009))
                  THEN P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES 
             WHEN P.PI_VISTA BETWEEN 0 AND 1.01 THEN P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES 
            ELSE 0 END) AS MIX_COMPETITIVENESS,
    */
    SUM(P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES) as  VISITS_TOTAL,
    SUM(CASE WHEN PI_VISTA > 0 OR PL1P_PRICING_CURRENT_WINNING_STRATEGY IN ('MATCH_EXT','ADJUSTED_PRICE','COMPETITORS_SCORE_A_PRICE','ADJUSTED_SCORE_A_PRICE') 
             THEN P.PL1P_TIME_IN_STATUS_PERC*V.VISITS_TOTALES ELSE 0 END)  as  VISITS_TOTAL_ONLY_PI

  FROM meli-bi-data.WHOWNER.DM_PL1P_PRICING_PRICE_INDEX AS P
  LEFT JOIN `meli-bi-data.SBOX_PRICING1P.VISITAS_TOTALES` AS V ON P.TIM_DAY = V.TIM_DAY AND P.SIT_SITE_ID = V.SIT_SITE_ID AND P.ITE_ITEM_ID = V.ITE_ITEM_ID
  --LEFT JOIN `WHOWNER.LK_ITE_ATTRIBUTE_VALUES` IAV ON P.SIT_SITE_ID = IAV.SIT_SITE_ID AND P.ITE_ITEM_ID = IAV.ITE_ITEM_ID
  LEFT JOIN (
    SELECT *
    , ROW_NUMBER() OVER(PARTITION BY SIT_SITE_ID, ITE_ITEM_ID ORDER BY CASE WHEN SAP_VENDOR_ESTIMATED  IS NOT NULL THEN 1 ELSE 0 END DESC ) AS RW 
    FROM meli-bi-data.SBOX_PLANNING_1P.VW_BRANDS_VENDORS_ITEM_ID 
    QUALIFY RW =1
    ) VD ON VD.SIT_SITE_ID = P.SIT_SITE_ID AND VD.ITE_ITEM_ID = P.ITE_ITEM_ID
LEFT JOIN WHOWNER.LK_ITE_ITEMS LKI ON P.SIT_SITE_ID = LKI.SIT_SITE_ID AND P.ITE_ITEM_ID = LKI.ITE_ITEM_ID

  --LEFT JOIN meli-bi-data.WHOWNER.LK_PL1P_ITE_VAR_SKU SK ON P.SIT_SITE_ID = SK.SIT_SITE_ID AND P.ITE_ITEM_ID = SK.ITE_ITEM_ID
  WHERE P.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO')
    --AND VERTICAL IN ('CE', 'CPG', 'APP & SPORTS')
    --AND P.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL','BEAUTY')
  AND P.TIM_DAY >= '2022-12-01'
  GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15