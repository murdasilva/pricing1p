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
    PCB.DOM_DOMAIN_AGG1,
    PCB.DOM_DOMAIN_AGG2,
    PCB.ITE_ATT_BRAND,
    PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
    CASE WHEN PO.ITE_SITE_CURRENT_PRICE >= PO.PROFITABILITY_PRICE_VALUE THEN 1 ELSE 0 END as FLAG_PRICE_ABOVE_FLOOR,
  --- AGREGO EL CAMPO DE BLACKLIST/WHITELIST
    CASE 
      WHEN (PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH')) THEN "Blacklist"
      WHEN (PCB.BLACKLIST_TYPE IN ('MARKDOWN','NONE')) THEN "Whitelist"
      ELSE "REVISAR" END AS ADOPTION,
    PCB.BLACKLIST_TYPE,
    PCB.PL1P_TIME_IN_STATUS_PERC,
     --COALESCE(
      --  SUM(PCB.PL1P_TIME_IN_STATUS_PERC) / NULLIF(SUM(PCB.PL1P_TIME_IN_STATUS_PERC) OVER (PARTITION BY PCB.SIT_SITE_ID, PCB.ITE_ITEM_ID, PCB.TIM_DAY), 0), 
       -- 0
    ---) AS PL1P_TIME_IN_STATUS_POND,

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
  LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_PRICING_OPPS` PO ON PCB.PL1P_PRICING_STATUS_FROM_DTTM = PO.LAST_UPDATED_FROM_DTTM AND PCB.SIT_SITE_ID = PO.SIT_SITE_ID AND PCB.ITE_ITEM_ID = PO.ITE_ITEM_ID
  WHERE PCB.SIT_SITE_ID in ('MLB','MLM','MLC','MLA','MCO')
    AND PCB.TIM_DAY BETWEEN '2022-01-01' AND '2023-08-07'
    AND PCB.ITE_ITEM_STATUS = 'ACTIVE'
    AND PCB.CUS_NICKNAME NOT LIKE 'PL%'
    --GROUP BY ALL
  )
UNION ALL  
-- A partir del 8/8/2023 Manual se considera como Automatico por el RO Nodo estandar absorbente  // RO todas las verticales menos CE 1/8/2023, RO Completo 8/8/2023 y se habilita MLA dado que durante semana 33 se encendieron algunos ítems de Apparel
  (SELECT
    PCB.TIM_DAY,
    PCB.SIT_SITE_ID,
    PCB.ITE_ITEM_ID,
    PCB.VERTICAL,
    PCB.DOM_DOMAIN_AGG1,
    PCB.DOM_DOMAIN_AGG2,
    PCB.ITE_ATT_BRAND,
    PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
        CASE WHEN PO.ITE_SITE_CURRENT_PRICE >= PO.PROFITABILITY_PRICE_VALUE THEN 1 ELSE 0 END as FLAG_PRICE_ABOVE_FLOOR,

    CASE 
      WHEN (PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH')) THEN "Blacklist" 
      WHEN (PCB.BLACKLIST_TYPE IN ('MARKDOWN','NONE')) THEN "Whitelist" ELSE "REVISAR" END AS ADOPTION,
    PCB.BLACKLIST_TYPE,
    PCB.PL1P_TIME_IN_STATUS_PERC,
       --  COALESCE(
       -- SUM(PCB.PL1P_TIME_IN_STATUS_PERC) / NULLIF(SUM(PCB.PL1P_TIME_IN_STATUS_PERC) OVER (PARTITION BY PCB.SIT_SITE_ID, PCB.ITE_ITEM_ID, PCB.TIM_DAY), 0), 
  --      0
  --  ) AS PL1P_TIME_IN_STATUS_POND,

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
   LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_PRICING_OPPS` PO ON PCB.PL1P_PRICING_STATUS_FROM_DTTM = PO.LAST_UPDATED_FROM_DTTM AND PCB.SIT_SITE_ID = PO.SIT_SITE_ID AND PCB.ITE_ITEM_ID = PO.ITE_ITEM_ID
  WHERE PCB.SIT_SITE_ID in ('MLB','MLM','MLC','MLA','MCO')
    AND PCB.TIM_DAY BETWEEN '2023-08-08' AND '2024-09-14'
    AND PCB.ITE_ITEM_STATUS = 'ACTIVE'
    AND PCB.CUS_NICKNAME NOT LIKE 'PL%'
    --GROUP BY ALL
  )

UNION ALL  
-- A partir de 2024-09-15 consideramos exceções de Deals sem contracogs que passam a ser considerados como automáticos
  (SELECT
    PCB.TIM_DAY,
    PCB.SIT_SITE_ID,
    PCB.ITE_ITEM_ID,
    PCB.VERTICAL,
    PCB.DOM_DOMAIN_AGG1,
    PCB.DOM_DOMAIN_AGG2,
    PCB.ITE_ATT_BRAND,
    PCB.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
        CASE WHEN PO.ITE_SITE_CURRENT_PRICE >= PO.PROFITABILITY_PRICE_VALUE THEN 1 ELSE 0 END as FLAG_PRICE_ABOVE_FLOOR,

    CASE 
      WHEN (PCB.BLACKLIST_TYPE IN ('ALL','BOTH','MATCH')) THEN "Blacklist" 
      WHEN (PCB.BLACKLIST_TYPE IN ('MARKDOWN','NONE')) THEN "Whitelist" ELSE "REVISAR" END AS ADOPTION,
    PCB.BLACKLIST_TYPE,
    PCB.PL1P_TIME_IN_STATUS_PERC,
       --  COALESCE(
       -- SUM(PCB.PL1P_TIME_IN_STATUS_PERC) / NULLIF(SUM(PCB.PL1P_TIME_IN_STATUS_PERC) OVER (PARTITION BY PCB.SIT_SITE_ID, PCB.ITE_ITEM_ID, PCB.TIM_DAY), 0), 
  --      0
  --  ) AS PL1P_TIME_IN_STATUS_POND,

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
  LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_PRICING_OPPS` PO ON PCB.PL1P_PRICING_STATUS_FROM_DTTM = PO.LAST_UPDATED_FROM_DTTM AND PCB.SIT_SITE_ID = PO.SIT_SITE_ID AND PCB.ITE_ITEM_ID = PO.ITE_ITEM_ID
  LEFT JOIN DEALS_EXCEPTIONS_1 DE1 ON PCB.SIT_SITE_ID = DE1.SIT_SITE_ID AND PCB.ITE_ITEM_ID = DE1.ITE_ITEM_ID
  WHERE PCB.SIT_SITE_ID in ('MLB','MLM','MLC','MLA','MCO')
    AND PCB.TIM_DAY BETWEEN '2024-09-15' AND CURRENT_DATE
    AND PCB.ITE_ITEM_STATUS = 'ACTIVE'
    AND PCB.CUS_NICKNAME NOT LIKE 'PL%'
    --GROUP BY ALL
  )
)


,IVENTORY_METRICS AS (
  SELECT TIM_DAY
    ,SIT_SITE_ID
    ,ITE_ITEM_ID
    ,SUM(TSI) AS TSI
    ,SUM(TGMV_LC) AS TGMV
    ,SUM(TSI_VALUED_ESTIMATED_LC) AS TSI_VALUED_ESTIMATED
    ,SUM(AGREEMENT_AMOUNT_ESTIMATED_LC) AS AGREEMENT_AMOUNT_ESTIMATED_LC
    ,SUM(TCOUPON_AMOUNT_IVA_DEDUCTED_LC) AS TCOUPON_AMOUNT_IVA_DEDUCTED_LC
  FROM `meli-bi-data.WHOWNER.DM_MKP_PL1P_INVENTORY_METRICS` IM
  WHERE TIM_DAY >= '2022-01-01'
  GROUP BY 1,2,3
)


,PARTE_FINAL AS 
(
SELECT *
  ,CASE
    WHEN BLACKLIST_TYPE IN ('ALL','BOTH','MATCH') AND TYPE_PRICING_AUTOMATIC_MANUAL = 'Manual Pricing' THEN 'Base pricing'
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
    ELSE PL1P_PRICING_CURRENT_WINNING_STRATEGY END AS PL1P_PRICING_CURRENT_WINNING_STRATEGY_2
  FROM
  (SELECT 
    E.TIM_DAY,
    CAST(
        CASE 
            WHEN EXTRACT(WEEK FROM E.TIM_DAY) < 10 
            THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY), '0', EXTRACT(WEEK FROM E.TIM_DAY)) 
            ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY), EXTRACT(WEEK FROM E.TIM_DAY)) 
        END AS INT
    ) AS WEEK,
    CAST(
        CASE 
            WHEN EXTRACT(MONTH FROM E.TIM_DAY) < 10 
            THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY), '0', EXTRACT(MONTH FROM E.TIM_DAY)) 
            ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY), EXTRACT(MONTH FROM E.TIM_DAY)) 
        END AS INT
    ) AS MONTH,
    CAST(
        CASE 
            WHEN EXTRACT(QUARTER FROM E.TIM_DAY) < 10 
            THEN CONCAT(EXTRACT(YEAR FROM E.TIM_DAY), '0', EXTRACT(QUARTER FROM E.TIM_DAY)) 
            ELSE CONCAT(EXTRACT(YEAR FROM E.TIM_DAY), EXTRACT(QUARTER FROM E.TIM_DAY)) 
        END AS INT
    ) AS QUARTER,
    E.SIT_SITE_ID,
    E.ITE_ITEM_ID,
    E.VERTICAL,
    CASE 
        WHEN E.DOM_DOMAIN_AGG1 IN ('FOOTWEAR', 'APPAREL ACCESSORIES', 'APPAREL') THEN 'APPAREL'
        WHEN E.DOM_DOMAIN_AGG1 IN ('BEAUTY') THEN 'BEAUTY'
        ELSE NULL 
    END AS SUBVERTICAL,
    E.DOM_DOMAIN_AGG1,
    E.DOM_DOMAIN_AGG2,
    E.FLAG_PRICE_ABOVE_FLOOR,
    E.ITE_ATT_BRAND,
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
    SUM(E.PL1P_TIME_IN_STATUS_PERC * V.VISITS_TOTALES) AS VISITS_TOTAL,
    SUM(E.PL1P_TIME_IN_STATUS_PERC * IM.TSI) AS TSI,
    SUM(E.PL1P_TIME_IN_STATUS_PERC * IM.TGMV) AS TGMV,
    SUM(E.PL1P_TIME_IN_STATUS_PERC * IM.TSI_VALUED_ESTIMATED) AS TSI_VALUED_ESTIMATED,
    SUM(E.PL1P_TIME_IN_STATUS_PERC * IM.AGREEMENT_AMOUNT_ESTIMATED_LC) AS AGREEMENT_AMOUNT_ESTIMATED_LC,
    SUM(E.PL1P_TIME_IN_STATUS_PERC * IM.TCOUPON_AMOUNT_IVA_DEDUCTED_LC) AS TCOUPON_AMOUNT_IVA_DEDUCTED_LC,
    SUM(E.PL1P_TIME_IN_STATUS_PERC) AS PL1P_TIME_IN_STATUS_PERC,
 --SUM(E.PL1P_TIME_IN_STATUS_PERC) OVER (PARTITION BY E.SIT_SITE_ID, E.ITE_ITEM_ID, E.TIM_DAY) AS PL1P_TIME_IN_STATUS_DENO

FROM EVOLUCION E
LEFT JOIN SBOX_PRICING1P.VISITAS_TOTALES V 
    ON V.TIM_DAY = E.TIM_DAY 
    AND V.SIT_SITE_ID = E.SIT_SITE_ID 
    AND V.ITE_ITEM_ID = E.ITE_ITEM_ID 
LEFT JOIN IVENTORY_METRICS IM 
    ON E.TIM_DAY = IM.TIM_DAY 
    AND E.SIT_SITE_ID = IM.SIT_SITE_ID 
    AND E.ITE_ITEM_ID = IM.ITE_ITEM_ID
WHERE E.TIM_DAY >= '2022-01-01'
GROUP BY 
    E.TIM_DAY,
    WEEK,
    MONTH,
    QUARTER,
    E.SIT_SITE_ID,
    E.ITE_ITEM_ID,
    E.TYPE_DETAIL_PRICING,
    E.VERTICAL,
    SUBVERTICAL,
    E.DOM_DOMAIN_AGG1,
    E.DOM_DOMAIN_AGG2,
    E.FLAG_PRICE_ABOVE_FLOOR,
    E.ITE_ATT_BRAND,
    E.PL1P_PRICING_CURRENT_WINNING_STRATEGY,
    E.ADOPTION,
    E.BLACKLIST_TYPE,
    TYPE_PRICING,
    TYPE_PRICING_AUTOMATIC_MANUAL)

)

SELECT *
  ,SAFE_DIVIDE(PL1P_TIME_IN_STATUS_PERC , SUM(PL1P_TIME_IN_STATUS_PERC)  OVER (PARTITION BY SIT_SITE_ID, ITE_ITEM_ID, TIM_DAY) ) AS PL1P_TIME_IN_STATUS_POND
FROM (
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
      --ADICIONAR AS DEMAIS COLUNAS
      TYPE_PRICING_AUTOMATIC_MANUAL,
      PL1P_PRICING_CURRENT_WINNING_STRATEGY_2,
      ITE_ITEM_SUPERMARKET_FLG,
      ITE_ITEM_SCHEDULED_FLG,
      FLAG_PRICE_ABOVE_FLOOR,
          CASE 
        WHEN  PL1P_PRICING_CURRENT_WINNING_STRATEGY_2 = 'Deals & Promos' AND TYPE_DETAIL_PRICING IN ('3-Tier wo/CCogs' , '4-DoD & Flash wo/CCogs','3a-Tier wo/CCogs exception','4a-DoD & Lightning wo/CCogs exception' )   AND FLAG_PRICE_ABOVE_FLOOR = 1 THEN  'Deal arriba del piso w/o CCOGS'
        WHEN  PL1P_PRICING_CURRENT_WINNING_STRATEGY_2 = 'Deals & Promos' AND TYPE_DETAIL_PRICING IN ('3-Tier wo/CCogs' , '4-DoD & Flash wo/CCogs','3a-Tier wo/CCogs exception','4a-DoD & Lightning wo/CCogs exception' )   AND FLAG_PRICE_ABOVE_FLOOR = 0 THEN  'Deal abajo del piso w/o CCOGS'
        WHEN  PL1P_PRICING_CURRENT_WINNING_STRATEGY_2 = 'Deals & Promos' AND TYPE_DETAIL_PRICING IN ('5-Tier w/CCogs','6-DoD & Lightning w/CCogs' )   AND FLAG_PRICE_ABOVE_FLOOR = 1 THEN  'Deal arriba del piso w CCOGS'
        WHEN  PL1P_PRICING_CURRENT_WINNING_STRATEGY_2 = 'Deals & Promos' AND TYPE_DETAIL_PRICING IN ('5-Tier w/CCogs','6-DoD & Lightning w/CCogs' )   AND FLAG_PRICE_ABOVE_FLOOR = 0 THEN  'Deal abajo del piso w CCOGS'
        ELSE 'Non Deal'
        END AS DEAL_DETAIL,

      SUM(VISITS_TOTAL) as VISITS_TOTAL,
      SUM(VISITS_AUTOMATICL_PRICE) AS VISITS_AUTOMATICL_PRICE,
      SUM(TSI) AS TSI,
      SUM(TGMV) AS TGMV,
      SUM(TSI_VALUED_ESTIMATED) AS TSI_VALUED_ESTIMATED,
      SUM(AGREEMENT_AMOUNT_ESTIMATED_LC) AS AGREEMENT_AMOUNT_ESTIMATED_LC,
      SUM(TCOUPON_AMOUNT_IVA_DEDUCTED_LC) AS TCOUPON_AMOUNT_IVA_DEDUCTED_LC ,
    --  SUM(PL1P_TIME_IN_STATUS_POND) PL1P_TIME_IN_STATUS_POND ,
      SUM(P.PL1P_TIME_IN_STATUS_PERC)  AS PL1P_TIME_IN_STATUS_PERC
  FROM PARTE_FINAL AS P
  LEFT JOIN     
      (SELECT *
        , ROW_NUMBER() OVER(PARTITION BY SIT_SITE_ID, ITE_ITEM_ID ORDER BY CASE WHEN SAP_VENDOR_ESTIMATED  IS NOT NULL THEN 1 ELSE 0 END DESC ) AS RW 
        FROM meli-bi-data.SBOX_PLANNING_1P.VW_BRANDS_VENDORS_ITEM_ID 
        QUALIFY RW =1
        ) VD ON P.SIT_SITE_ID = VD.SIT_SITE_ID AND P.ITE_ITEM_ID = VD.ITE_ITEM_ID
  LEFT JOIN WHOWNER.LK_ITE_ITEMS LKI ON P.SIT_SITE_ID = LKI.SIT_SITE_ID AND P.ITE_ITEM_ID = LKI.ITE_ITEM_ID
  GROUP BY ALL
)
