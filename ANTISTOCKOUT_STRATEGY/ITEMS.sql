WITH LUMPSUM AS (
  SELECT
    A.SIT_SITE_ID,
    C.ITE_ITEM_ID,
    'LUMPSUM' AS DEAL_TYPE
  FROM `meli-bi-data.WHOWNER.BT_PL1P_AGREEMENTS_TRANSACTIONAL` AS A
  LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_AGREEMENTS` AS B ON A.AGREE_ID = B.AGREE_ID
  LEFT JOIN `meli-bi-data.WHOWNER.LK_PL1P_ITE_VAR_SKU` AS C ON lpad(A.SAP_SKU,18,'0') = C.ITE_ITEM_SAP_SKU
  WHERE 1=1    
    AND A.TIM_DAY BETWEEN DATE_ADD(CURRENT_DATE(), INTERVAL -1 DAY) AND CURRENT_DATE()
    AND (B.AGREE_SUB_TYPE_NAME not in ('MELI_PLUS','PURCHASING_BONUS') OR B.AGREE_SUB_TYPE_NAME IS NULL)
    AND A.AGREE_ID IS NOT NULL
    AND B.AGREE_TYPE != 'SELL'
  GROUP BY ALL
)

,FINANCIAL_DATA AS ( 

    SELECT 
        --CAST(UPDATED_DATE AS DATE) AS TIM_DAY
        --ITEM_ID,
        LEFT(ITEM_ID,3) AS SIT_SITE_ID,
        CAST(RIGHT(ITEM_ID, LENGTH(ITEM_ID)-3) AS BIGINT) as ITE_ITEM_ID,
        COST,
        CCOGS,
        SIT_SITE_IVA,
        FINANCIAL_COST,
        INSTALLMENTS,
        PPM_PROFIT_FLOOR,
        PPM_CALCULATED_FLOOR_PRICE,
        ROW_NUMBER() OVER (PARTITION BY ITEM_ID ORDER BY UPDATED_DATE DESC) AS RW,
    FROM WHOWNER.LK_PL1P_UE_MATERIAL_SUMMARIZATION_HISTORY
    WHERE UPDATED_DATE >= '2024-01-01'
    QUALIFY RW = 1
)

,SELLOUT AS (
  SELECT SIT_SITE_ID
    ,ITE_ITEM_ID
    ,'SELLOUT' AS DEAL_TYPE
  FROM FINANCIAL_DATA
  WHERE COALESCE(CCOGS,0) > 0

)

,CCOGS_LIST AS (
  SELECT SIT_SITE_ID,
    ITE_ITEM_ID,
    ARRAY_AGG(DEAL_TYPE) AS DEAL_TYPE
  FROM (
    SELECT SIT_SITE_ID
      ,ITE_ITEM_ID
      ,DEAL_TYPE
    FROM SELLOUT

    UNION ALL


    SELECT SIT_SITE_ID
      ,ITE_ITEM_ID
      ,DEAL_TYPE
    FROM LUMPSUM
  )
  GROUP BY ALL
)

,ITEMS AS (

    SELECT SIT_SITE_ID,
    ITE_ITEM_ID,
    ITE_ITEM_STATUS,
    ITE_ITEM_INVENTORY_ID,
    ITE_ITEM_QUANTITY_AVAILABLE,
    ITE_ITEM_CATALOG_LISTING_FLG,
    ITE_ITEM_RELATIONS

    FROM WHOWNER.LK_ITE_ITEMS
    WHERE ITE_ITEM_PARTY_TYPE_ID = '1P'
    AND SIT_SITE_ID  IN ('MLB','MLA','MLC','MLM','MCO')
    AND ITE_ITEM_STATUS IN ('active','paused')
    AND ITE_ITEM_QUANTITY_AVAILABLE > 0
    AND ITE_ITEM_INVENTORY_ID IS NOT NULL
    AND IS_TEST = FALSE

)

, CATALOGO_HERMANADO AS (
    SELECT SIT_SITE_ID,
        ITE_ITEM_ID,
        FROM (
        SELECT 
            SIT_SITE_ID,
            ITE_ITEM_ID,
            R.ITEM_ID,
        FROM ITEMS, UNNEST(ITE_ITEM_RELATIONS) R
        WHERE ITE_ITEM_CATALOG_LISTING_FLG = TRUE
        )
        GROUP BY 1,2
)

SELECT I.SIT_SITE_ID,
    I.ITE_ITEM_ID,
    I.ITE_ITEM_STATUS,
    I.ITE_ITEM_INVENTORY_ID,
    I.ITE_ITEM_QUANTITY_AVAILABLE,
    I.ITE_ITEM_CATALOG_LISTING_FLG,
    CASE WHEN C.ITE_ITEM_ID IS NULL AND ITE_ITEM_CATALOG_LISTING_FLG = TRUE THEN 1 ELSE 0 END AS FLAG_CATALOGO_NO_HERMANADO,
    CASE WHEN L.ITE_ITEM_ID IS NOT  NULL THEN 1 ELSE 0 END AS FLAG_ACUERDO
FROM ITEMS I
LEFT JOIN CATALOGO_HERMANADO C ON I.SIT_SITE_ID = C.SIT_SITE_ID AND I.ITE_ITEM_ID = C.ITE_ITEM_ID
LEFT JOIN CCOGS_LIST AS L ON I.SIT_SITE_ID = L.SIT_SITE_ID AND I.ITE_ITEM_ID = L.ITE_ITEM_ID
WHERE 
    (ITE_ITEM_CATALOG_LISTING_FLG = FALSE
    OR
    (ITE_ITEM_CATALOG_LISTING_FLG = TRUE AND C.ITE_ITEM_ID IS NULL) -- ITEMS DE CATALOGO SEM HERMANADOS
    ) 
