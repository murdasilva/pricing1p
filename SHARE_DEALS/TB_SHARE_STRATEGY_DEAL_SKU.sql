-- AP Final | version 20240327 (correct AGREE_SUB_TYPE_NAME filtering) 

WITH ACUERDOS AS (
  SELECT
    A.TIM_DAY,
    A.SIT_SITE_ID,
    C.ITE_ITEM_ID,
    CONCAT(A.SIT_SITE_ID,C.ITE_ITEM_ID) AS ITEM_ID
  FROM `meli-bi-data.WHOWNER.BT_PL1P_AGREEMENTS_TRANSACTIONAL` AS A
  LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_AGREEMENTS` AS B ON A.AGREE_ID = B.AGREE_ID
  LEFT JOIN `meli-bi-data.WHOWNER.DM_MKP_PL1P_INVENTORY_METRICS` AS C ON lpad(A.SAP_SKU,18,'0') = C.ITE_ITEM_SAP_SKU AND A.TIM_DAY = C.TIM_DAY
  WHERE 1=1    
    AND A.TIM_DAY BETWEEN '2022-01-01' AND CURRENT_DATE()
    AND (B.AGREE_SUB_TYPE_NAME not in ('MELI_PLUS','PURCHASING_BONUS') OR B.AGREE_SUB_TYPE_NAME IS NULL)
    AND A.AGREE_ID IS NOT NULL
  GROUP BY 1,2,3,4
  ORDER BY 1,2,3,4 DESC
),

EVOLUCION AS ( 
  (SELECT
    PCB.TIM_DAY,
    PCB.SIT_SITE_ID,
    PCB.ITE_ITEM_ID,
    PCB.VERTICAL,
    PCB.DOM_DOMAIN_AGG1,
    PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
  --- AGREGO EL CAMPO DE BLACKLIST/WHITELIST
    CASE 
      WHEN (PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH')) THEN "Blacklist"
      WHEN (PCB.BLACKLIST_TYPE IN ('MARKDOWN','NONE')) THEN "Whitelist"
      ELSE "REVISAR" END AS ADOPTION,
    PCB.BLACKLIST_TYPE,
    PCB.PL1P_TIME_IN_STATUS_PERC,
    PCB.PL1P_PRICING_STATUS_FROM_DTTM,
    CASE 
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC') AND PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH') THEN '1-Blacklist'
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'MANUAL'THEN '2-Manual Strategy'
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NULL THEN '3-Tier wo/CCogs' -- Sin CCogs      
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NULL THEN '4-DoD & Lightning wo/CCogs' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NOT NULL THEN '5-Tier w/CCogs' -- Con CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NOT NULL THEN '6-DoD & Lightning w/CCogs' -- Con CCogs   
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC') 
           AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY 
            IN ('MATCH_EXT',
                'COMPETITORS_SCORE_A_PRICE',
                'ADJUSTED_PRICE',
                'ADJUSTED_SCORE_A_PRICE',
                'MARKDOWN',
                'HISTORIC_PRICE',
                'FALLBACK_BASE_PRICE',
                'TARGET_PRICE',
                'FALLBACK_PRICE',
                'BUYBOX_WINNER_PRICE',
                'PROFITABILITY_PRICE',
                'LIST_PRICE',
                'BASE_PRICE',
                'NO_MATCH',
                'FALLBACK_MSRP',
                'LIST_PRICE_CAP',
                'FALLBACK_STD') THEN '7-Automatic Strategies'
    WHEN PCB.SIT_SITE_ID IN ('MLA') THEN '1-Blacklist' ELSE 'REVISAR' END AS TYPE_DETAIL_PRICING
  FROM EXPLOTACION.1P_PRICING_PERFORMANCE PCB
  LEFT JOIN ACUERDOS AS A ON PCB.ITE_ITEM_ID = A.ITE_ITEM_ID AND PCB.SIT_SITE_ID = A.SIT_SITE_ID AND PCB.TIM_DAY = A.TIM_DAY
  WHERE PCB.SIT_SITE_ID in ('MLB','MLM','MLC','MLA')
    AND PCB.TIM_DAY BETWEEN '2022-01-01' AND '2023-08-07'
    AND PCB.ITE_ITEM_STATUS = 'ACTIVE'
    AND PCB.CUS_NICKNAME NOT LIKE 'PL%'
  )
UNION ALL  
-- A partir del 8/8/2023 Manual se considera como Automatico por el RO Nodo estandar absorbente  // RO todas las verticales menos CE 1/8/2023, RO Completo 8/8/2023 y se habilita MLA dado que durante semana 33 se encendieron algunos ítems de Apparel
  (SELECT
    PCB.TIM_DAY,
    PCB.SIT_SITE_ID,
    PCB.ITE_ITEM_ID,
    PCB.VERTICAL,
    PCB.DOM_DOMAIN_AGG1,
    PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
    CASE 
      WHEN (PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH')) THEN "Blacklist" 
      WHEN (PCB.BLACKLIST_TYPE IN ('MARKDOWN','NONE')) THEN "Whitelist" ELSE "REVISAR" END AS ADOPTION,
    PCB.BLACKLIST_TYPE,
    PCB.PL1P_TIME_IN_STATUS_PERC,
    PCB.PL1P_PRICING_STATUS_FROM_DTTM,
    CASE 
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA') AND PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH') THEN '1-Blacklist'
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA') 
           AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY 
           IN ('MATCH_EXT',
              'COMPETITORS_SCORE_A_PRICE',
              'ADJUSTED_PRICE',
              'ADJUSTED_SCORE_A_PRICE',
              'MARKDOWN',
              'HISTORIC_PRICE',
              'FALLBACK_BASE_PRICE',
              'TARGET_PRICE',
              'FALLBACK_PRICE',
              'BUYBOX_WINNER_PRICE',
              'PROFITABILITY_PRICE',
              'LIST_PRICE',
              'BASE_PRICE',
              'NO_MATCH',
              'FALLBACK_MSRP',
              'LIST_PRICE_CAP',
              'FALLBACK_STD',
              'MANUAL') ---CHANGE
              THEN '7-Automatic Strategies'
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NULL THEN '3-Tier wo/CCogs' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NULL THEN '4-DoD & Lightning wo/CCogs' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NOT NULL THEN '5-Tier w/CCogs' -- Con CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NOT NULL THEN '6-DoD & Lightning w/CCogs' -- Con CCogs      
    ELSE 'REVISAR' END AS TYPE_DETAIL_PRICING
  FROM EXPLOTACION.1P_PRICING_PERFORMANCE PCB
  LEFT JOIN ACUERDOS AS A ON PCB.ITE_ITEM_ID = A.ITE_ITEM_ID AND PCB.SIT_SITE_ID = A.SIT_SITE_ID AND PCB.TIM_DAY = A.TIM_DAY
  WHERE PCB.SIT_SITE_ID in ('MLB','MLM','MLC','MLA')
    AND PCB.TIM_DAY BETWEEN '2023-08-08' AND CURRENT_DATE()
    AND PCB.ITE_ITEM_STATUS = 'ACTIVE'
    AND PCB.CUS_NICKNAME NOT LIKE 'PL%'
  )
)

,FT AS (

SELECT 
  E.TIM_DAY,
  CAST(CASE WHEN EXTRACT(ISOWEEK FROM E.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),'0',EXTRACT(ISOWEEK FROM E.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),EXTRACT(ISOWEEK FROM E.TIM_DAY)) END AS INT) AS WEEK,
  CAST(CASE WHEN EXTRACT(MONTH FROM E.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),'0',EXTRACT(MONTH FROM E.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),EXTRACT(MONTH FROM E.TIM_DAY)) END AS INT) AS MONTH,
  CAST(CASE WHEN EXTRACT(QUARTER FROM E.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),'0',EXTRACT(QUARTER FROM E.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),EXTRACT(QUARTER FROM E.TIM_DAY)) END AS INT) AS QUARTER,
  E.SIT_SITE_ID,
  E.VERTICAL,
  CASE 
        WHEN E.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL') THEN 'APPAREL'
        WHEN E.DOM_DOMAIN_AGG1 IN ('BEAUTY') THEN 'BEAUTY'
        ELSE NULL END AS SUBVERTICAL,

  E.DOM_DOMAIN_AGG1,
  E.ITE_ITEM_ID,
  E.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
  E.ADOPTION,
  E.BLACKLIST_TYPE,
  E.TYPE_DETAIL_PRICING, 
  CASE 
    WHEN TYPE_DETAIL_PRICING IN ('1-Blacklist') THEN '1-Blacklist'
    WHEN TYPE_DETAIL_PRICING IN ('2-Manual Strategy') THEN '2-Manual'
    WHEN TYPE_DETAIL_PRICING IN ('3-Tier wo/CCogs') THEN '3-Tier wo/CCogs'
    WHEN TYPE_DETAIL_PRICING IN ('4-DoD & Lightning wo/CCogs') THEN '4-DoD & Flash wo/CCogs'
    WHEN TYPE_DETAIL_PRICING IN ('5-Tier w/CCogs','6-DoD & Lightning w/CCogs','7-Automatic Strategies') THEN 'Automatic' 
    ELSE 'REVIEW' END AS TYPE_PRICING,
  CASE 
    WHEN TYPE_DETAIL_PRICING IN ('1-Blacklist','2-Manual Strategy','3-Tier wo/CCogs','4-DoD & Lightning wo/CCogs') THEN 'Manual Pricing'
    WHEN TYPE_DETAIL_PRICING IN ('5-Tier w/CCogs','6-DoD & Lightning w/CCogs','7-Automatic Strategies') THEN 'Automatic Pricing' 
    ELSE 'REVIEW' END AS  TYPE_PRICING_AUTOMATIC_MANUAL,
  E.PL1P_TIME_IN_STATUS_PERC,
  E.PL1P_TIME_IN_STATUS_PERC AS PL1P_TIME_IN_STATUS_PERC,
  E.PL1P_TIME_IN_STATUS_PERC/SUM(E.PL1P_TIME_IN_STATUS_PERC) OVER (PARTITION BY E.SIT_SITE_ID, E.ITE_ITEM_ID, E.TIM_DAY) AS PL1P_TIME_IN_STATUS_POND,
  CASE WHEN PO.ITE_SITE_CURRENT_PRICE >= PO.PROFITABILITY_PRICE_VALUE THEN 1 ELSE 0 END as FLAG_PRICE_ABOVE_FLOOR
  
FROM EVOLUCION E
LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_PRICING_OPPS` PO ON E.PL1P_PRICING_STATUS_FROM_DTTM = PO.LAST_UPDATED_FROM_DTTM AND E.SIT_SITE_ID = PO.SIT_SITE_ID AND E.ITE_ITEM_ID = PO.ITE_ITEM_ID
--LEFT JOIN SBOX_PRICING1P.VISITAS_TOTALES V ON V.TIM_DAY = E.TIM_DAY AND V.SIT_SITE_ID = E.SIT_SITE_ID AND V.ITE_ITEM_ID = E.ITE_ITEM_ID 
WHERE E.TIM_DAY >= '2022-01-01' -- DEJO SOLO LOS MESES QUE SIRVEN PARA EL GRÁFICO DE LA MONTHLY
  AND PL1P_TIME_IN_STATUS_PERC >0
    --AND E.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL','BEAUTY')
--GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
)

SELECT TIM_DAY
    ,WEEK
    ,MONTH
    ,QUARTER
    ,A.SIT_SITE_ID
    ,ITE_ITEM_SAP_SKU
    ,TYPE_PRICING_AUTOMATIC_MANUAL
    ,FLAG_PRICE_ABOVE_FLOOR
    ,CASE 
      WHEN  PL1P_PRICING_CURRENT_WINNING_STRATEGY_2 = 'Deals & Promos' AND TYPE_PRICING_AUTOMATIC_MANUAL = 'Manual Pricing'   AND FLAG_PRICE_ABOVE_FLOOR = 1 THEN  'Deal arriba del piso w/o CCOGS'
      WHEN  PL1P_PRICING_CURRENT_WINNING_STRATEGY_2 = 'Deals & Promos' AND TYPE_PRICING_AUTOMATIC_MANUAL = 'Manual Pricing'   AND FLAG_PRICE_ABOVE_FLOOR = 0 THEN  'Deal abajo del piso w/o CCOGS'
      WHEN  PL1P_PRICING_CURRENT_WINNING_STRATEGY_2 = 'Deals & Promos' AND TYPE_PRICING_AUTOMATIC_MANUAL = 'Automatic Pricing'   AND FLAG_PRICE_ABOVE_FLOOR = 1 THEN  'Deal arriba del piso w CCOGS'
      WHEN  PL1P_PRICING_CURRENT_WINNING_STRATEGY_2 = 'Deals & Promos' AND TYPE_PRICING_AUTOMATIC_MANUAL = 'Automatic Pricing'   AND FLAG_PRICE_ABOVE_FLOOR = 0 THEN  'Deal abajo del piso w CCOGS'
      ELSE 'Non Deal'
      END AS DEAL_DETAIL

    ,PERC_TIME_IN_STATUS/(SUM(PERC_TIME_IN_STATUS) OVER( PARTITION BY TIM_DAY, A.SIT_SITE_ID , ITE_ITEM_SAP_SKU) ) AS PERC_TIME_IN_STATUS
FROM (
  SELECT TIM_DAY
    ,WEEK
    ,MONTH
    ,QUARTER
    ,FT.SIT_SITE_ID
    ,FT.ITE_ITEM_ID
    ,CASE
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'ADJUSTED_PRICE' THEN 'External match'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'ADJUSTED_SCORE_A_PRICE' THEN 'External match'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'BUYBOX_WINNER_PRICE' THEN 'BB Winner'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'COMPETITORS_SCORE_A_PRICE' THEN 'External match'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' THEN 'Deals & Promos'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'FALLBACK_PRICE' THEN 'Base pricing'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'FALLBACK_STD' THEN 'Base pricing'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'HISTORIC_PRICE' THEN 'Base pricing'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'LIST_PRICE' THEN 'Base pricing'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'MANUAL' THEN 'Base pricing'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'MARKDOWN' THEN 'Markdown'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'MATCH_EXT' THEN 'External match'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROFITABILITY_PRICE' THEN 'Profit floor'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' THEN 'Deals & Promos'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'TARGET_PRICE' THEN 'Base pricing'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'BASE_PRICE' THEN 'Base pricing'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'FALLBACK_MSRP' THEN 'Base pricing'
      WHEN PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'LIST_PRICE_CAP' THEN 'Base pricing'
      ELSE PL1P_PRICING_CURRENT_WINNING_STRATEGY END AS PL1P_PRICING_CURRENT_WINNING_STRATEGY_2,
    TYPE_PRICING_AUTOMATIC_MANUAL,
    FLAG_PRICE_ABOVE_FLOOR,
    SUM(PL1P_TIME_IN_STATUS_POND) AS PERC_TIME_IN_STATUS
  FROM FT
  GROUP BY 1,2,3,4,5,6,7,8,9
  --ORDER BY 1,2,3,4,5,6,7,8,9
) AS A

LEFT JOIN WHOWNER.LK_PL1P_ITE_VAR_SKU AS SK ON A.SIT_SITE_ID = SK.SIT_SITE_ID AND A.ITE_ITEM_ID = SK.ITE_ITEM_ID