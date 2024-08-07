-- # IF, Con visitas totales
-- #20231201 -  new status table  
-- 20240321 - New date filters applied (2024+)

WITH 

LK_ITEMS AS (
SELECT SIT_SITE_ID, ITE_ITEM_ID
FROM WHOWNER.LK_ITE_ITEMS
WHERE ITE_ITEM_PARTY_TYPE_ID = '1P'
),

STATUS_ITEMS AS (
  SELECT 
    ITE_ITEM_ID,
    SIT_SITE_ID,
    EXTRACT(DATE FROM DATETIME_INS) AS TIM_DAY,
    MIN(ITE_ITEM_STATUS) as STATUS
  FROM `meli-bi-data.COMPETENCIA.STATUS_ITEMS_1P_HOUR`
  WHERE EXTRACT(DATE FROM DATETIME_INS) >= '2022-12-01' 
    AND ITE_ITEM_STATUS = 'active'
  GROUP BY 1,2,3
 ),


--ETAPA 1 hasta el 30/11/2023
HOUND_1_RAW AS ( 
  SELECT
    CONCAT(H.SIT_SITE_ID,H.ITE_ITEM_ID) AS key_SITE_ID_ITE_ITEM_ID,
    H.SIT_SITE_ID,
    H.ITE_ITEM_ID,
    H.COMP_SITE_ID,
    H.COMP_SELLER_SCORE,
    CAST(CASE WHEN EXTRACT(WEEK FROM SCRAP_DATE) < 10 THEN CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),'0',EXTRACT(WEEK FROM SCRAP_DATE)) ELSE CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),EXTRACT(WEEK FROM SCRAP_DATE)) END AS INT) AS WEEK,  
    CAST(CASE WHEN EXTRACT(MONTH FROM SCRAP_DATE) < 10 THEN CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),'0',EXTRACT(MONTH FROM SCRAP_DATE)) ELSE CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),EXTRACT(MONTH FROM SCRAP_DATE)) END AS INT) AS MONTH,  
    CAST(CASE WHEN EXTRACT(QUARTER FROM SCRAP_DATE) < 10 THEN CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),'0',EXTRACT(QUARTER FROM SCRAP_DATE)) ELSE CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),EXTRACT(QUARTER FROM SCRAP_DATE)) END AS INT) AS QUARTER,
    SCRAP_DATE,
    MIN(COMP_OFFERS.COMP_ITEM_PRICE) AS COMP_ITEM_PRICE
  FROM `meli-bi-data.COMPETENCIA.LK_COMP_HOUND_ITEMS` H, UNNEST(COMP_OFFERS) AS COMP_OFFERS
  --INNER JOIN LK_ITEMS I ON H.SIT_SITE_ID = I.SIT_SITE_ID AND H.ITE_ITEM_ID = CAST(I.ITE_ITEM_ID AS STRING) 
  WHERE 
    COMP_SELLER_SCORE	IN ('A','B')
    --AND COMP_FLAG_FIRST_PARTY = 'true'
    AND H.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') 
    AND SCRAP_DATE BETWEEN '2022-12-01' and '2023-11-30'
    --AND SCRAP_DATE BETWEEN '2022-10-01' and '2023-11-30'
  GROUP BY 1,2,3,4,5,6,7,8,9
  --ORDER BY 3
),



HOUND_1 AS (
  SELECT H.*
  FROM HOUND_1_RAW H
  INNER JOIN LK_ITEMS I 
    ON H.SIT_SITE_ID = I.SIT_SITE_ID AND H.ITE_ITEM_ID = CAST(I.ITE_ITEM_ID AS STRING)
),


COVERAGE_1 AS (
SELECT 
  CAST(CASE WHEN EXTRACT(WEEK FROM C.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM C.TIM_DAY),'0',EXTRACT(WEEK FROM C.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM C.TIM_DAY),EXTRACT(WEEK FROM C.TIM_DAY)) END AS INT) AS WEEK,  
  CAST(CASE WHEN EXTRACT(MONTH FROM C.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM C.TIM_DAY),'0',EXTRACT(MONTH FROM C.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM C.TIM_DAY),EXTRACT(MONTH FROM C.TIM_DAY)) END AS INT) AS MONTH,  
  CAST(CASE WHEN EXTRACT(QUARTER FROM C.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM C.TIM_DAY),'0',EXTRACT(QUARTER FROM C.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM C.TIM_DAY),EXTRACT(QUARTER FROM C.TIM_DAY)) END AS INT) AS QUARTER,
  C.TIM_DAY,
  C.SIT_SITE_ID AS SITE,
  C.VERTICAL,
CASE 
    WHEN C.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL') THEN 'APPAREL'
    WHEN C.DOM_DOMAIN_AGG1 IN ('BEAUTY') THEN 'BEAUTY'
    ELSE NULL END AS SUBVERTICAL,
  C.DOM_DOMAIN_AGG1,
  C.DOM_DOMAIN_AGG2,
  C.ITE_ATT_BRAND,
  C.ITE_ITEM_ID,
  C.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
  C.COVERAGE_TYPE,
  CASE WHEN C.BLACKLIST_TYPE in ('NONE', 'MARKDOWN') then 'WL' else 'BL' end as BL_TYPE,
  CASE WHEN C.COVERAGE_TYPE = 'Net' then 'con_cobertura' else 'sin_cobertura' end as COBERTURA_TYPE,
  CASE WHEN C.COVERAGE_TYPE = 'Net' THEN V.VISITS_TOTALES*PL1P_TIME_IN_STATUS_PERC ELSE 0 END AS SUMA_VISITS_TOTALES_COB,
  V.VISITS_TOTALES*PL1P_TIME_IN_STATUS_PERC AS VISITAS_TOTALES,	
FROM WHOWNER.DM_PL1P_PRICING_COVERAGE AS C
LEFT JOIN `meli-bi-data.SBOX_PRICING1P.VISITAS_TOTALES` AS V 
  ON C.TIM_DAY = V.TIM_DAY AND C.SIT_SITE_ID = V.SIT_SITE_ID AND C.ITE_ITEM_ID = V.ITE_ITEM_ID
WHERE
  C.ITE_ITEM_STATUS = 'ACTIVE'
  AND C.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO')
  AND C.TIM_DAY BETWEEN '2022-12-01' and '2023-11-30'
  --AND C.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL','BEAUTY')
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17

),


-- ETAPA 2 a partir del 1/12/2023
HOUND_2_RAW AS ( -- #NOTE: Busco los competidores en hound, de las ultimas semanas
  SELECT
    CONCAT(H.SIT_SITE_ID,H.ITE_ITEM_ID) AS key_SITE_ID_ITE_ITEM_ID,
    H.SIT_SITE_ID,
    H.ITE_ITEM_ID,
    H.COMP_SITE_ID,
    H.COMP_SELLER_SCORE,
    CAST(CASE WHEN EXTRACT(WEEK FROM SCRAP_DATE) < 10 THEN CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),'0',EXTRACT(WEEK FROM SCRAP_DATE)) ELSE CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),EXTRACT(WEEK FROM SCRAP_DATE)) END AS INT) AS WEEK,  
    CAST(CASE WHEN EXTRACT(MONTH FROM SCRAP_DATE) < 10 THEN CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),'0',EXTRACT(MONTH FROM SCRAP_DATE)) ELSE CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),EXTRACT(MONTH FROM SCRAP_DATE)) END AS INT) AS MONTH,  
    CAST(CASE WHEN EXTRACT(QUARTER FROM SCRAP_DATE) < 10 THEN CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),'0',EXTRACT(QUARTER FROM SCRAP_DATE)) ELSE CONCAT(EXTRACT(YEAR FROM SCRAP_DATE),EXTRACT(QUARTER FROM SCRAP_DATE)) END AS INT) AS QUARTER,
    SCRAP_DATE,
    MIN(COMP_OFFERS.COMP_ITEM_PRICE) AS COMP_ITEM_PRICE
  FROM `meli-bi-data.COMPETENCIA.LK_COMP_HOUND_ITEMS` H, UNNEST(COMP_OFFERS) AS COMP_OFFERS
  --INNER JOIN STATUS_ITEMS S 
    --ON  H.SCRAP_DATE = S.TIM_DAY AND CAST(S.ITE_ITEM_ID AS STRING) = H.ITE_ITEM_ID AND S.SIT_SITE_ID = H.SIT_SITE_ID --AND S.STATUS = 'active'
  WHERE 
    COMP_SELLER_SCORE	IN ('A','B')
    AND H.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') 
    AND SCRAP_DATE >= '2023-12-01' 
  GROUP BY 1,2,3,4,5,6,7,8,9
 -- ORDER BY 3
),

HOUND_2 AS (
  SELECT H2.*
  FROM HOUND_2_RAW H2
  INNER JOIN STATUS_ITEMS S 
    ON  H2.SCRAP_DATE = S.TIM_DAY AND CAST(S.ITE_ITEM_ID AS STRING) = H2.ITE_ITEM_ID AND S.SIT_SITE_ID = H2.SIT_SITE_ID 
),

COVERAGE_2 AS (
SELECT 
  CAST(CASE WHEN EXTRACT(WEEK FROM C2.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM C2.TIM_DAY),'0',EXTRACT(WEEK FROM C2.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM C2.TIM_DAY),EXTRACT(WEEK FROM C2.TIM_DAY)) END AS INT) AS WEEK,  
  CAST(CASE WHEN EXTRACT(MONTH FROM C2.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM C2.TIM_DAY),'0',EXTRACT(MONTH FROM C2.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM C2.TIM_DAY),EXTRACT(MONTH FROM C2.TIM_DAY)) END AS INT) AS MONTH,  
  CAST(CASE WHEN EXTRACT(QUARTER FROM C2.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM C2.TIM_DAY),'0',EXTRACT(QUARTER FROM C2.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM C2.TIM_DAY),EXTRACT(QUARTER FROM C2.TIM_DAY)) END AS INT) AS QUARTER,
  C2.TIM_DAY,
  C2.SIT_SITE_ID AS SITE,
  C2.VERTICAL,
  CASE 
        WHEN C2.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL') THEN 'APPAREL'
        WHEN C2.DOM_DOMAIN_AGG1 IN ('BEAUTY') THEN 'BEAUTY'
        ELSE NULL END AS SUBVERTICAL,
  C2.DOM_DOMAIN_AGG1,
  C2.DOM_DOMAIN_AGG2,
  C2.ITE_ATT_BRAND,
  C2.ITE_ITEM_ID,
  C2.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
  C2.COVERAGE_TYPE,
  case when C2.BLACKLIST_TYPE in ('NONE', 'MARKDOWN') then 'WL' else 'BL' end as BL_TYPE,
  case when C2.COVERAGE_TYPE = 'Net' then 'con_cobertura' else 'sin_cobertura' end as COBERTURA_TYPE,
  CASE WHEN C2.COVERAGE_TYPE = 'Net' THEN V.VISITS_TOTALES*PL1P_TIME_IN_STATUS_PERC ELSE 0 END AS SUMA_VISITS_TOTALES_COB,
  V.VISITS_TOTALES*PL1P_TIME_IN_STATUS_PERC AS VISITAS_TOTALES,	
FROM WHOWNER.DM_PL1P_PRICING_COVERAGE AS C2
LEFT JOIN `meli-bi-data.SBOX_PRICING1P.VISITAS_TOTALES` AS V 
  ON C2.TIM_DAY = V.TIM_DAY AND C2.SIT_SITE_ID = V.SIT_SITE_ID AND C2.ITE_ITEM_ID = V.ITE_ITEM_ID
INNER JOIN STATUS_ITEMS S 
  ON  C2.TIM_DAY = S.TIM_DAY AND C2.ITE_ITEM_ID = S.ITE_ITEM_ID AND S.SIT_SITE_ID = C2.SIT_SITE_ID --AND STATUS = 'active'
WHERE
  C2.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') 
  AND C2.TIM_DAY >= '2023-12-01' 
  --AND C2.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL','BEAUTY')

GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
)

,FINAL_TABLE AS 
  --Union de etapas
  (
  (
    SELECT 
      C.*,
      COUNT(DISTINCT H.COMP_SITE_ID) AS CANT_COMPETIDORES,
      CASE WHEN COUNT(DISTINCT H.COMP_SITE_ID) > 0 THEN C.SUMA_VISITS_TOTALES_COB ELSE 0 END AS VISITS_1IF,
      CASE WHEN COUNT(DISTINCT H.COMP_SITE_ID) > 1 THEN C.SUMA_VISITS_TOTALES_COB ELSE 0 END AS VISITS_2IF,
      CASE WHEN COUNT(DISTINCT H.COMP_SITE_ID) > 2 THEN C.SUMA_VISITS_TOTALES_COB ELSE 0 END AS VISITS_3IF,
    FROM COVERAGE_1 AS C
    LEFT JOIN HOUND_1 AS H
      ON CONCAT(C.SITE,C.ITE_ITEM_ID) = H.key_SITE_ID_ITE_ITEM_ID AND H.SCRAP_DATE = C.TIM_DAY 
    GROUP BY 1,2,3,4,5,6,7,8,9,9,10,11,12,13,14,15,16,17
  )
  UNION ALL

  (
    SELECT 
      C2.*,
      COUNT(DISTINCT H2.COMP_SITE_ID) AS CANT_COMPETIDORES,
      CASE WHEN COUNT(DISTINCT H2.COMP_SITE_ID) > 0 THEN C2.SUMA_VISITS_TOTALES_COB ELSE 0 END AS VISITS_1IF,
      CASE WHEN COUNT(DISTINCT H2.COMP_SITE_ID) > 1 THEN C2.SUMA_VISITS_TOTALES_COB ELSE 0 END AS VISITS_2IF,
      CASE WHEN COUNT(DISTINCT H2.COMP_SITE_ID) > 2 THEN C2.SUMA_VISITS_TOTALES_COB ELSE 0 END AS VISITS_3IF,
  FROM COVERAGE_2 AS C2
    LEFT JOIN HOUND_2 AS H2
      ON CONCAT(C2.SITE,C2.ITE_ITEM_ID) = H2.key_SITE_ID_ITE_ITEM_ID AND H2.SCRAP_DATE = C2.TIM_DAY 
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
  )
  )

SELECT 
  FT.TIM_DAY,
  FT.WEEK,
  FT.MONTH,
  FT.QUARTER,
  FT.SITE AS SIT_SITE_ID,
  FT.ITE_ITEM_ID,
  FT.VERTICAL,
  FT.SUBVERTICAL,
  FT.DOM_DOMAIN_AGG1,
  FT.DOM_DOMAIN_AGG2,
  FT.ITE_ATT_BRAND,
  VD.SAP_VENDOR_ESTIMATED,
  SUM(VISITS_1IF) AS VISITS_1IF,
  SUM(VISITS_2IF) AS VISITS_2IF,
  SUM(VISITS_3IF) AS VISITS_3IF,
  SUM(VISITAS_TOTALES) as VISITAS_TOTALES
FROM FINAL_TABLE FT
  LEFT JOIN     
  (SELECT *
    , ROW_NUMBER() OVER(PARTITION BY SIT_SITE_ID, ITE_ITEM_ID ORDER BY CASE WHEN SAP_VENDOR_ESTIMATED  IS NOT NULL THEN 1 ELSE 0 END DESC ) AS RW 
    FROM meli-bi-data.SBOX_PLANNING_1P.VW_BRANDS_VENDORS_ITEM_ID 
    QUALIFY RW =1
    ) VD ON FT.SITE = VD.SIT_SITE_ID AND FT.ITE_ITEM_ID = VD.ITE_ITEM_ID
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12