SELECT
    t1.SIT_SITE_ID,
    t1.ITE_ITEM_ID,
    t3.ITE_ITEM_SAP_SKU,
    t1.ITE_ITEM_STATUS,
    t1.CUS_CUST_ID_SEL as seller_id,
    CONCAT(t1.SIT_SITE_ID, t1.CAT_CATEG_ID) as category_id,
    t1.ITE_ITEM_DOM_DOMAIN_ID as domain_id
FROM `meli-bi-data.WHOWNER.LK_ITE_ITEMS` as t1
INNER JOIN
    `meli-bi-data.WHOWNER.LK_CUS_CUSTOMERS_DATA` as t2 ON t1.CUS_CUST_ID_SEL=t2.CUS_CUST_ID, unnest(CUS_INTERNAL_TAGS) int_tags
LEFT JOIN
    `meli-bi-data.WHOWNER.LK_PL1P_ITE_VAR_SKU` as t3 ON t1.SIT_SITE_ID = t3.SIT_SITE_ID and t1.ITE_ITEM_ID=t3.ITE_ITEM_ID
WHERE
    int_tags in ('first_party')
    and t1.ITE_ITEM_STATUS in('active', 'paused', 'under_review')