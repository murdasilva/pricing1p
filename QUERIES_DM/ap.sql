WITH ACUERDOS AS (
  SELECT
    A.TIM_DAY,
    A.SIT_SITE_ID,
    C.ITE_ITEM_ID,
    CONCAT(A.SIT_SITE_ID,C.ITE_ITEM_ID) AS ITEM_ID
  FROM `meli-bi-data.WHOWNER.BT_PL1P_AGREEMENTS_TRANSACTIONAL` AS A
  LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_AGREEMENTS` AS B ON A.AGREE_ID = B.AGREE_ID
  LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_ITE_VAR_SKU` AS C ON lpad(A.SAP_SKU,18,'0') = C.ITE_ITEM_SAP_SKU
  WHERE 1=1    
    AND A.TIM_DAY BETWEEN '2022-01-01' AND CURRENT_DATE()
    AND (B.AGREE_SUB_TYPE_NAME not in ('MELI_PLUS','PURCHASING_BONUS') OR B.AGREE_SUB_TYPE_NAME IS NULL)
    AND A.AGREE_ID IS NOT NULL
  GROUP BY 1,2,3,4
  ORDER BY 1,2,3,4 DESC
),

-- ITEMS_DETAILS AS (

--   SELECT IT.SIT_SITE_ID
--     ,IT.ITE_ITEM_ID
--     ,IT.ITE_ITEM_PARTY_TYPE_ID
--     ,DOM.VERTICAL
--     ,DOM.DOM_DOMAIN_AGG1
--     ,DOM.DOM_DOMAIN_AGG2
--     ,V.SAP_VENDOR_ESTIMATED
--   FROM WHOWNER.LK_ITE_ITEMS AS IT
--   LEFT JOIN `meli-bi-data.WHOWNER.LK_ITE_ITEM_DOMAINS` DOM ON IT.SIT_SITE_ID = DOM.SIT_SITE_ID AND IT.ITE_ITEM_ID = DOM.ITE_ITEM_ID
--   LEFT JOIN ( -- Subquery para buscar o vendor na IM
--     SELECT SIT_SITE_ID
--       ,ITE_ITEM_ID
--       ,SAP_VENDOR_ESTIMATED
--       ,ROW_NUMBER() OVER( PARTITION BY SIT_SITE_ID, ITE_ITEM_ID ORDER BY TIM_DAY DESC) AS RW
--     FROM WHOWNER.DM_MKP_PL1P_INVENTORY_METRICS
--     WHERE TIM_DAY >= DATE_ADD( CURRENT_DATE, INTERVAL -90 DAY)
--     QUALIFY RW=1
--   ) AS V ON IT.SIT_SITE_ID = V.SIT_SITE_ID AND IT.ITE_ITEM_ID = V.ITE_ITEM_ID
--   WHERE IT.ITE_ITEM_PARTY_TYPE_ID = '1P'
-- ),

-- DEALS_EXCEPTIONS_1 AS (

--   SELECT ID.SIT_SITE_ID
--     ,ID.ITE_ITEM_ID
--   FROM ITEMS_DETAILS ID
--   WHERE  (
--     (ID.SIT_SITE_ID IN ('MLB', 'MLM', 'MLC') AND ID.DOM_DOMAIN_AGG1 IN ('FOOTWEAR','APPAREL ACCESSORIES','APPAREL')) -- APPAREL MLB , MLM e MLC
--     OR
--     (ID.SIT_SITE_ID IN ('MLA','MCO')) -- ALL MLA AND MCO
--     OR
--     (ID.SIT_SITE_ID = 'MLC' AND ID.VERTICAL = 'CE' AND ID.SAP_VENDOR_ESTIMATED = 'APPLE CHILE COMERCIAL LIMITADA') -- Apple MLC
--     OR
--     (ID.SIT_SITE_ID = 'MLC' AND ID.VERTICAL = 'APPAREL & BEAUTY' AND ID.SAP_VENDOR_ESTIMATED = 'GRETA CHILE SPA') -- Greta MLC
--     OR
--     (ID.SIT_SITE_ID = 'MLC' AND ID.VERTICAL = 'APPAREL & BEAUTY' AND ID.SAP_VENDOR_ESTIMATED = 'DIFFUPAR CHILE SPA') -- DIFFUPAR MLC
--     OR
--     (ID.SIT_SITE_ID = 'MLC' AND ID.VERTICAL = 'CE' AND ID.SAP_VENDOR_ESTIMATED = 'INTCOMEX SA') -- INTCOMEX MLC
--     OR
--     (ID.SIT_SITE_ID = 'MLC' AND ID.VERTICAL = 'CE' AND ID.SAP_VENDOR_ESTIMATED = 'INGRAM MICRO CHILE SA') -- INGRAM MLC
--   )

-- ),


DEALS_EXCEPTIONS_1 as (
  SELECT
    CAST(NULL AS STRING) AS SIT_SITE_ID,
    CAST(NULL AS INT) AS ITE_ITEM_ID,
),


EVOLUCION AS ( 
  (SELECT
    PCB.TIM_DAY,
    PCB.SIT_SITE_ID,
    PCB.ITE_ITEM_ID,
    PCB.VERTICAL,
    PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
  --- AGREGO EL CAMPO DE BLACKLIST/WHITELIST
    CASE 
      WHEN (PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH')) THEN "Blacklist"
      WHEN (PCB.BLACKLIST_TYPE IN ('MARKDOWN','NONE')) THEN "Whitelist"
      ELSE "REVISAR" END AS ADOPTION,
    PCB.BLACKLIST_TYPE,
    PCB.PL1P_TIME_IN_STATUS_PERC,
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
  FROM WHOWNER.DM_PL1P_PRICING_PERFORMANCE PCB
  LEFT JOIN ACUERDOS AS A ON PCB.ITE_ITEM_ID = A.ITE_ITEM_ID AND PCB.SIT_SITE_ID = A.SIT_SITE_ID AND PCB.TIM_DAY = A.TIM_DAY
  WHERE PCB.SIT_SITE_ID in ('MLB','MLM','MLC','MLA','MCO')
    AND PCB.TIM_DAY BETWEEN '2022-10-01' AND '2023-08-07'
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
    PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
    CASE 
      WHEN (PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH')) THEN "Blacklist" 
      WHEN (PCB.BLACKLIST_TYPE IN ('MARKDOWN','NONE')) THEN "Whitelist" ELSE "REVISAR" END AS ADOPTION,
    PCB.BLACKLIST_TYPE,
    PCB.PL1P_TIME_IN_STATUS_PERC,
    CASE 
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH') THEN '1-Blacklist'
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') 
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
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NULL THEN '3-Tier wo/CCogs' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NULL THEN '4-DoD & Lightning wo/CCogs' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NOT NULL THEN '5-Tier w/CCogs' -- Con CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NOT NULL THEN '6-DoD & Lightning w/CCogs' -- Con CCogs      
    ELSE 'REVISAR' END AS TYPE_DETAIL_PRICING
  FROM WHOWNER.DM_PL1P_PRICING_PERFORMANCE PCB
  LEFT JOIN ACUERDOS AS A ON PCB.ITE_ITEM_ID = A.ITE_ITEM_ID AND PCB.SIT_SITE_ID = A.SIT_SITE_ID AND PCB.TIM_DAY = A.TIM_DAY
  WHERE PCB.SIT_SITE_ID in ('MLB','MLM','MLC','MLA','MCO')
    AND PCB.TIM_DAY BETWEEN '2023-08-08' AND '2024-09-14'
    AND PCB.ITE_ITEM_STATUS = 'ACTIVE'
    AND PCB.CUS_NICKNAME NOT LIKE 'PL%'
  )
  UNION ALL  
 -- A partir de 2024-09-15 consideramos exceções de Deals sem contracogs que passam a ser considerados como automáticos
  (SELECT
    PCB.TIM_DAY,
    PCB.SIT_SITE_ID,
    PCB.ITE_ITEM_ID,
    PCB.VERTICAL,
    PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
    CASE 
      WHEN (PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH')) THEN "Blacklist" 
      WHEN (PCB.BLACKLIST_TYPE IN ('MARKDOWN','NONE')) THEN "Whitelist" ELSE "REVISAR" END AS ADOPTION,
    PCB.BLACKLIST_TYPE,
    PCB.PL1P_TIME_IN_STATUS_PERC,
    CASE 
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH') THEN '1-Blacklist'
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') 
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
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NULL AND DE1.ITE_ITEM_ID IS NULL THEN '3-Tier wo/CCogs' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NULL AND DE1.ITE_ITEM_ID IS NOT NULL THEN '3a-Tier wo/CCogs exception' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NULL AND DE1.ITE_ITEM_ID IS NULL THEN '4-DoD & Lightning wo/CCogs' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NULL AND DE1.ITE_ITEM_ID IS NOT NULL THEN '4a-DoD & Lightning wo/CCogs exception' -- Sin CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'DEAL' AND A.ITEM_ID IS NOT NULL THEN '5-Tier w/CCogs' -- Con CCogs
      WHEN PCB.SIT_SITE_ID IN ('MLB','MLM','MLC','MLA','MCO') AND PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY = 'PROMO' AND A.ITEM_ID IS NOT NULL THEN '6-DoD & Lightning w/CCogs' -- Con CCogs  
    ELSE 'REVISAR' END AS TYPE_DETAIL_PRICING
  FROM WHOWNER.DM_PL1P_PRICING_PERFORMANCE PCB
  LEFT JOIN ACUERDOS AS A ON PCB.ITE_ITEM_ID = A.ITE_ITEM_ID AND PCB.SIT_SITE_ID = A.SIT_SITE_ID AND PCB.TIM_DAY = A.TIM_DAY
  LEFT JOIN DEALS_EXCEPTIONS_1 DE1 ON PCB.SIT_SITE_ID = DE1.SIT_SITE_ID AND PCB.ITE_ITEM_ID = DE1.ITE_ITEM_ID
  WHERE PCB.SIT_SITE_ID in ('MLB','MLM','MLC','MLA','MCO')
    AND PCB.TIM_DAY BETWEEN '2024-09-15' AND CURRENT_DATE
    AND PCB.ITE_ITEM_STATUS = 'ACTIVE'
    AND PCB.CUS_NICKNAME NOT LIKE 'PL%'
  )
)


SELECT 
  E.TIM_DAY,
  CAST(CASE WHEN EXTRACT(ISOWEEK FROM E.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),'0',EXTRACT(ISOWEEK FROM E.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),EXTRACT(ISOWEEK FROM E.TIM_DAY)) END AS INT) AS WEEK,
  CAST(CASE WHEN EXTRACT(MONTH FROM E.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),'0',EXTRACT(MONTH FROM E.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),EXTRACT(MONTH FROM E.TIM_DAY)) END AS INT) AS MONTH,
  CAST(CASE WHEN EXTRACT(QUARTER FROM E.TIM_DAY) < 10 THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),'0',EXTRACT(QUARTER FROM E.TIM_DAY)) ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY),EXTRACT(QUARTER FROM E.TIM_DAY)) END AS INT) AS QUARTER,
  E.SIT_SITE_ID,
  E.VERTICAL,
  E.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
  E.ADOPTION,
  E.BLACKLIST_TYPE,
  E.TYPE_DETAIL_PRICING, 
  CASE 
    WHEN TYPE_DETAIL_PRICING IN ('1-Blacklist') THEN '1-Blacklist'
    WHEN TYPE_DETAIL_PRICING IN ('2-Manual Strategy') THEN '2-Manual'
    WHEN TYPE_DETAIL_PRICING IN ('3-Tier wo/CCogs') THEN '3-Tier wo/CCogs'
    WHEN TYPE_DETAIL_PRICING IN ('4-DoD & Lightning wo/CCogs') THEN '4-DoD & Flash wo/CCogs'
    WHEN TYPE_DETAIL_PRICING IN ('5-Tier w/CCogs','6-DoD & Lightning w/CCogs','7-Automatic Strategies','3a-Tier wo/CCogs exception','4a-DoD & Lightning wo/CCogs exception') THEN 'Automatic' 
    ELSE 'REVIEW' END AS TYPE_PRICING,
  CASE 
    WHEN TYPE_DETAIL_PRICING IN ('1-Blacklist','2-Manual Strategy','3-Tier wo/CCogs','4-DoD & Lightning wo/CCogs') THEN 'Manual Pricing'
    WHEN TYPE_DETAIL_PRICING IN ('5-Tier w/CCogs','6-DoD & Lightning w/CCogs','7-Automatic Strategies','3a-Tier wo/CCogs exception','4a-DoD & Lightning wo/CCogs exception') THEN 'Automatic Pricing' 
    ELSE 'REVIEW' END AS  TYPE_PRICING_AUTOMATIC_MANUAL,
  SUM(CASE WHEN TYPE_DETAIL_PRICING IN ('5-Tier w/CCogs','6-DoD & Lightning w/CCogs','7-Automatic Strategies','3a-Tier wo/CCogs exception','4a-DoD & Lightning wo/CCogs exception') THEN PL1P_TIME_IN_STATUS_PERC*VISITS_TOTALES ELSE 0 END) AS VISITS_AUTOMATICL_PRICE,
  SUM(PL1P_TIME_IN_STATUS_PERC*VISITS_TOTALES) AS VISITS_TOTAL
FROM EVOLUCION E
LEFT JOIN SBOX_PRICING1P.VISITAS_TOTALES V ON V.TIM_DAY = E.TIM_DAY AND V.SIT_SITE_ID = E.SIT_SITE_ID AND V.ITE_ITEM_ID = E.ITE_ITEM_ID 
WHERE E.TIM_DAY >= '2022-10-01' -- DEJO SOLO LOS MESES QUE SIRVEN PARA EL GRÁFICO DE LA MONTHLY
GROUP BY 1,2,3,4,5,6,7,8,9,10
