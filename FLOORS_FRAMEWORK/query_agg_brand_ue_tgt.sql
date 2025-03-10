WITH UE AS (

  SELECT SIT_SITE_ID,
    VERTICAL,
    DOM_DOMAIN_AGG2,
    ITE_ATT_BRAND,
    SUM(UE_CON_TGMV_AMT_LC) AS UE_CON_TGMV_AMT_LC,
    SUM(UE_MNG_VENDOR_MARGIN_AMT_LC) AS UE_MNG_VENDOR_MARGIN_AMT_LC,
    SUM(UE_MNG_VENDOR_MARGIN_ADJ_AMT_LC) AS UE_MNG_VENDOR_MARGIN_ADJ_AMT_LC,
    SUM(UE_MNG_VARIABLE_CONTRIBUTION_AMT_LC) AS UE_MNG_VARIABLE_CONTRIBUTION_AMT_LC,
    SUM(UE_MNG_VARIABLE_CONTRIBUTION_ADJ_AMT_LC) AS UE_MNG_VARIABLE_CONTRIBUTION_ADJ_AMT_LC,
    SUM(UE_MNG_DIRECT_CONTRIBUTION_AMT_LC) AS UE_MNG_DIRECT_CONTRIBUTION_AMT_LC,
    SUM(UE_MNG_DIRECT_CONTRIBUITION_ADJ_AMT_LC) AS UE_MNG_DIRECT_CONTRIBUITION_ADJ_AMT_LC,

  FROM `meli-bi-data.WHOWNER.BT_UE_OUTPUT_MANAGERIAL` M
  WHERE UE_PRC_BUSINESS_UNIT IN ('FIRST_PARTY')
    AND SIT_SITE_ID IN ('MLB','MLM','MLA','MCO', 'MLC')
    AND DATE_TRUNC(DIA_FINAL,MONTH) BETWEEN DATE_ADD(DATE_TRUNC(CURRENT_DATE,MONTH), INTERVAL -2 - 6 MONTH) AND DATE_ADD(DATE_TRUNC(CURRENT_DATE,MONTH), INTERVAL -2  MONTH)
  GROUP BY ALL
  )

  SELECT UE.*,
  T.TARGET_PRIORIZED
  FROM UE AS UE
  LEFT JOIN SBOX_PRICING1P.TARGETS_BPC_LANDED_AB_2025 T ON UE.SIT_SITE_ID = T.SITE AND UE.VERTICAL = T.VERTICAL AND T.MONTH = EXTRACT(YEAR FROM CURRENT_DATE)*100 + EXTRACT(MONTH FROM CURRENT_DATE)
