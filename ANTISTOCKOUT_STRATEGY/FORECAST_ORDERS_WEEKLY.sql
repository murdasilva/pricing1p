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

SELECT R.SIT_SITE_ID
  ,R.ITE_ITEM_INVENTORY_ID
  ,PERIOD
  ,SUM(Q_50) AS ORDERS_FORECAST
FROM RAW_FORECAST R
--LEFT JOIN WHOWNER.LK_PL1P_ITE_VAR_SKU SK ON R.SIT_SITE_ID = SK.SIT_SITE_ID AND R.SAP_SKU = SK.ITE_ITEM_SAP_SKU -- Pode se duplicar linhas quando um SKU esteja em mais de um item_id ; Ex: Tradicional e Catálogo
GROUP BY ALL

