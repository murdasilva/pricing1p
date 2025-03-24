min(
    max(
        min(
            COMP_PRICE_RIVAL,  //Prices affected by the floor
            PRICE_MELI         //Prices affected by the floor
        ),
        NEW_PROFITABILITY_PRICE
    ),
    DEAL_PRICE,      //Prices NOT affected by the floor
    PROMO_PRICE,     //Prices NOT affected by the floor
    MARKDOWN_PRICE   //Prices NOT affected by the floor
)



