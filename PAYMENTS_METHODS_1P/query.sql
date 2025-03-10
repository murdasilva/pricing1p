SELECT 
  SIT_SITE_ID,
  DIA_FINAL,
  VERTICAL,
  DOM_DOMAIN_AGG1,
  DOM_DOMAIN_AGG2,
  ITE_ATT_BRAND,
  SAP_VENDOR_ESTIMATED,
  CRT_PURCHASE_ID,
  ITE_ITEM_BULKY_FLG,
  ITE_ITEM_HB_FLG,
  ITE_ITEM_ID,
  ITE_ITEM_SKU_ID,
  ITE_ITEM_TITLE,
  ITE_SUPERMARKET_FLAG,
  ORD_ORDER_ID,
  ORD_STATUS,
  SAP_VENDOR_ESTIMATED,
  UE_FNC_PAY_COMBO_ID_DESC,
  UE_FNC_PAY_PM_TYPE_DESC,
  --UE_PRC_CBO_COMBO_ID,
  UE_FNC_INSTALLMENTS_QTY,
  SUM(UE_CON_TGMV_AMT_LC) AS UE_CON_TGMV_AMT_LC,
  SUM(UE_CON_TSI_AMT_LC) AS UE_CON_TSI_AMT_LC,
  SUM(UE_FNC_TPV_AMT_LC) AS UE_FNC_TPV_AMT_LC,
  SUM(UE_MNG_REVENUE_GROSS_AMT_LC) AS UE_MNG_REVENUE_GROSS_AMT_LC,
  SUM(UE_FNC_FINANCIAL_REVENUES_PSJ_AMT_LC) AS UE_FNC_FINANCIAL_REVENUES_PSJ_AMT_LC,
  SUM(UE_FNC_FINANCIAL_REVENUES_NO_PSJ_AMT_LC) AS UE_FNC_FINANCIAL_REVENUES_NO_PSJ_AMT_LC,
  SUM(UE_FNC_FINANCIAL_COST_NO_PSJ_AMT_LC) AS UE_FNC_FINANCIAL_COST_NO_PSJ_AMT_LC,
  SUM(UE_FNC_FINANCIAL_COST_PSJ_AMT_LC) AS UE_FNC_FINANCIAL_COST_PSJ_AMT_LC,
  --UE_MNG_ASP,


FROM meli-bi-data.WHOWNER.BT_UE_OUTPUT_MANAGERIAL UE
WHERE UE.SIT_SITE_ID IN ('MLA' , 'MLB' , 'MLC' , 'MLM' , 'MCO')
  AND UE_PRC_BUSINESS_UNIT = 'FIRST_PARTY'
  AND DATE_TRUNC(DIA_FINAL, MONTH) BETWEEN DATE_TRUNC(CURRENT_DATE, YEAR) AND DATE_ADD(DATE_TRUNC(CURRENT_DATE, MONTH), INTERVAL -2 MONTH)
  AND ORD_STATUS = 'paid'
  AND UE_FNC_PAY_COMBO_ID_DESC != 'HIBRIDOS'
GROUP BY ALL
ORDER BY DIA_FINAL DESC