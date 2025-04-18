WITH ITEMS AS (
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
        ITE_ITEM_RELATIONS,
        CASE WHEN COALESCE( ITE_ITEM_QUANTITY_AVAILABLE,0) < 1 THEN 1 ELSE 0 END AS FLAG_STOCKED_OUD,
        CASE WHEN ITE_ITEM_INVENTORY_ID IS NULL THEN 1 ELSE 0 END AS FLAG_NO_INVENTORY_ID,
        FROM WHOWNER.LK_ITE_ITEMS
        WHERE ITE_ITEM_PARTY_TYPE_ID = '1P'
        AND SIT_SITE_ID  IN ('MLB','MLA','MLC','MLM','MCO')
        AND ITE_ITEM_STATUS IN ('active','paused')
        --AND ITE_ITEM_QUANTITY_AVAILABLE > 0
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
        --FLAGS QUE EXCLUYEN ITEMS
        FLAG_STOCKED_OUD,
        FLAG_NO_INVENTORY_ID,
        CASE WHEN L.ITE_ITEM_ID IS NOT  NULL THEN 1 ELSE 0 END AS FLAG_ACUERDO
    FROM ITEMS I
    LEFT JOIN CATALOGO_HERMANADO C ON I.SIT_SITE_ID = C.SIT_SITE_ID AND I.ITE_ITEM_ID = C.ITE_ITEM_ID
    LEFT JOIN CCOGS_LIST AS L ON I.SIT_SITE_ID = L.SIT_SITE_ID AND I.ITE_ITEM_ID = L.ITE_ITEM_ID
    WHERE 
        (ITE_ITEM_CATALOG_LISTING_FLG = FALSE
        OR
        (ITE_ITEM_CATALOG_LISTING_FLG = TRUE AND C.ITE_ITEM_ID IS NULL) -- ITEMS DE CATALOGO SEM HERMANADOS
        ) 
)

,FORECAST AS (
    WITH RAW_FORECAST AS (

    SELECT 
        SIT_SITE_ID,
        MATERIAL_ID,
        SAP_SKU,
        ITE_ITEM_INVENTORY_ID,
        PERIOD,
        Q_50,
        CUTOFF,
        ROW_NUMBER() OVER( PARTITION BY SIT_SITE_ID, MATERIAL_ID, SAP_SKU, ITE_ITEM_INVENTORY_ID, PERIOD ORDER BY CUTOFF DESC) AS RW
    FROM WHOWNER.BT_1P_FORECAST
    WHERE 1=1 
        --AND SAP_SKU = "000000000001082927"
        --AND PERIOD = '2025-03-31'
        AND SKU_INACTIVE_FLG = FALSE
        AND PERIOD = DATE_TRUNC(PERIOD,ISOWEEK)
        AND PERIOD >= DATE_TRUNC(CURRENT_DATE,ISOWEEK)
        AND PERIOD <= DATE_ADD(DATE_TRUNC(CURRENT_DATE,ISOWEEK), INTERVAL +5 WEEK)
    QUALIFY RW = 1
    ORDER BY PERIOD DESC

    )

    , FORECAST_WEEKLY AS (
    SELECT R.SIT_SITE_ID
        ,R.ITE_ITEM_INVENTORY_ID
        ,PERIOD
        ,SUM(Q_50) AS ORDERS_FORECAST
    FROM RAW_FORECAST R
    --LEFT JOIN WHOWNER.LK_PL1P_ITE_VAR_SKU SK ON R.SIT_SITE_ID = SK.SIT_SITE_ID AND R.SAP_SKU = SK.ITE_ITEM_SAP_SKU -- Pode se duplicar linhas quando um SKU esteja em mais de um item_id ; Ex: Tradicional e Catálogo
    GROUP BY ALL
    )

    SELECT 
    F.SIT_SITE_ID,
    F.ITE_ITEM_INVENTORY_ID,
    F.PERIOD,
    --S.DOW,
    --S.HOUR_ORDER,
    DATE_ADD(PERIOD, INTERVAL MOD(DOW + 7  - 2, 7) DAY) AS TIM_DAY,
    --DATETIME_ADD( CAST(DATE_ADD(PERIOD, INTERVAL MOD(DOW + 7  - 2, 7) DAY) AS DATETIME), INTERVAL HOUR_ORDER HOUR) AS TIM_DATETIME,
    SUM(F.ORDERS_FORECAST* SHARE_TSI) AS ORDERS_FORECAST
    FROM FORECAST_WEEKLY F
    LEFT JOIN SBOX_PRICING1P.AUX_SAZONALIDADE_SEMANAL S ON F.SIT_SITE_ID = S.SIT_SITE_ID
    WHERE DATETIME_ADD( CAST(DATE_ADD(PERIOD, INTERVAL MOD(DOW + 7  - 2, 7) DAY) AS DATETIME), INTERVAL HOUR_ORDER HOUR) >= CURRENT_DATETIME
    GROUP BY ALL
)

,INBOUND AS (

    SELECT
    SIT_SITE_ID,
    INVENTORY_ID,
    CAST(INB_APPOINTMENT_DATETIME AS DATE) AS TIM_DAY,
    SUM(INB_QUANTITY) AS INB_QUANTITY
    FROM meli-bi-data.WHOWNER.BT_FBM_INBOUND_PANEL ING
    WHERE 1=1
    AND CAST(ING.INB_APPOINTMENT_DATETIME AS DATE) >= CURRENT_DATE
    AND INB_APPOINTMENT_DATETIME <= DATE_ADD(DATE_TRUNC(CURRENT_DATE,ISOWEEK), INTERVAL +5 WEEK)
    AND ING.INB_FLAG_REMOVAL = 0
    AND INB_STATUS  IN ('working','confirmed')
    AND ING.CUS_CUST_ID IN
        (
        449695682,
        794908452,
        557616860,
        527927603,
        466063253,
        481988724,
        550072427,
        550063615,
        550070308,
        480265022,
        768826009,
        451403353,
        480263032,
        608885756,
        608846165,
        1673696836,
        608887919,
        554328422,
        530380977,
        544457369
        )
    AND ING.SIT_SITE_ID IN ('MLB', 'MLM', 'MCO', 'MLC', 'MLA')
    GROUP BY ALL
    ORDER BY 1,2,3
)


, ESQUELETO AS (
    SELECT 
        T.TIM_DAY,
        I.* 
    FROM ITEMS I
    LEFT JOIN (
        SELECT
    *
    FROM 
        UNNEST(GENERATE_DATE_ARRAY(CURRENT_DATE, DATE_ADD(CURRENT_DATE, INTERVAL 29 DAY),INTERVAL 1 DAY)) AS TIM_DAY
        ) T ON 1=1
)


SELECT E.*
    ,COALESCE(F.ORDERS_FORECAST,0) AS ORDERS_FORECAST
    ,COALESCE(I.INB_QUANTITY,0) AS INB_QUANTITY
    ,COALESCE(I.INB_QUANTITY,0) - COALESCE(F.ORDERS_FORECAST,0)  AS DELTA_STOCK
    ,SUM(COALESCE(I.INB_QUANTITY,0) - COALESCE(F.ORDERS_FORECAST,0)) OVER (PARTITION BY E.SIT_SITE_ID, E.ITE_ITEM_ID ORDER BY E.TIM_DAY ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW ) AS DELTA_STOCK_ACCUM
    ,E.ITE_ITEM_QUANTITY_AVAILABLE + SUM(COALESCE(I.INB_QUANTITY,0) - COALESCE(F.ORDERS_FORECAST,0)) OVER (PARTITION BY E.SIT_SITE_ID, E.ITE_ITEM_ID ORDER BY E.TIM_DAY ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW ) AS STOCK_FORECAST
FROM ESQUELETO E 
LEFT JOIN FORECAST F ON E.TIM_DAY = F.TIM_DAY AND E.SIT_SITE_ID = F.SIT_SITE_ID AND E.ITE_ITEM_INVENTORY_ID = F.ITE_ITEM_INVENTORY_ID
LEFT JOIN INBOUND I ON E.TIM_DAY = I.TIM_DAY AND E.SIT_SITE_ID = I.SIT_SITE_ID AND E.ITE_ITEM_INVENTORY_ID = I.INVENTORY_ID
