SELECT
    CONCAT(t1.SIT_SITE_ID, t1.ITE_ITEM_ID) as ITEM_ID,
    MAX(t1.ORD_ITEM.UNIT_PRICE	) order_max_price
FROM `meli-bi-data.WHOWNER.BT_ORD_ORDERS` as t1
INNER JOIN
    `meli-bi-data.WHOWNER.LK_CUS_CUSTOMERS_DATA` as t2 ON t1.ord_seller.id=t2.CUS_CUST_ID, unnest(CUS_INTERNAL_TAGS) int_tags
WHERE
    int_tags in ('first_party')
    AND t1.ORD_CREATED_DT >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
    AND t1.ord_status not in ('cancelled', 'invalid', 'under_review')
GROUP BY
    1