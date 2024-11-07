 def process_info(self):
        active_offers = self.active_offers.copy()
        items_max_price = self.items_max_p.copy()
        df_sap = self.df_sap.copy()

        df_merged = active_offers.merge(df_sap, on=['SIT_SITE_ID', 'SAP_SKU'], how='left')
        df_merged = df_merged[~df_merged['ITE_ITEM_ID'].isnull()]
        df_merged['ITEM_ID'] = df_merged['SIT_SITE_ID'].map(str.strip) + df_merged['ITE_ITEM_ID'].astype(int).astype(str)

        df_filtered = df_merged.merge(items_max_price, on='ITEM_ID', how='left')
        df_filtered['PL1P_OFFE_SUGGESTED_PRICE_AMOUNT'] = df_filtered['PL1P_OFFE_SUGGESTED_PRICE_AMOUNT'].astype(float)
        df_filtered['PL1P_OFFE_UNIT_PRICE_AMOUNT'] = df_filtered['PL1P_OFFE_UNIT_PRICE_AMOUNT'].astype(float)

        prueba_skus = self.ggle.get_sheets_df('16htTXhDLOoVKvCSORXpIv8aKErNEfXqWu1Xt15s_A18', 'listado_skus')
        self.skus_especiales = [item[0].strip() for item in prueba_skus[1:]]
        datos_casos_especiales=df_filtered[df_filtered['SAP_SKU'].astype(str).isin( self.skus_especiales)].copy()

        # Rules of validation
        # PL1P_OFFE_SUGGESTED_PRICE_AMOUNT == LIST PRICE

        # List price greater than PO unit price
        mask_rules_1 = df_filtered['PL1P_OFFE_SUGGESTED_PRICE_AMOUNT'] > df_filtered['PL1P_OFFE_UNIT_PRICE_AMOUNT'] * 1.05

        # List price less than 2*(PO unit price)
        mask_rules_2 = df_filtered['PL1P_OFFE_SUGGESTED_PRICE_AMOUNT'] < 5 * df_filtered['PL1P_OFFE_UNIT_PRICE_AMOUNT']

        # List price greater than the max price of last 30 days
        mask_rules_3 = (df_filtered['PL1P_OFFE_SUGGESTED_PRICE_AMOUNT'] >= df_filtered['order_max_price'].astype(float)) | df_filtered['order_max_price'].isnull()

        df = df_filtered[mask_rules_1 & mask_rules_2 & mask_rules_3]

        df.sort_values(by='PL1P_OFFE_SUGGESTED_PRICE_AMOUNT', ascending=False, inplace=True)
        total = df.drop_duplicates(subset='ITEM_ID')
        result = pd.concat([total, datos_casos_especiales], ignore_index=True)
        # Get the max list price with validations
        result.sort_values(by='PL1P_OFFE_SUGGESTED_PRICE_AMOUNT', ascending=False, inplace=True)
        total = result.drop_duplicates(subset='ITEM_ID')
        total['LIST_PRICE'] = total['PL1P_OFFE_SUGGESTED_PRICE_AMOUNT']

        total.columns = map(str.lower, total.columns)
        total=total[~(total.domain_id.isin([' MLB-BABY_BOTTLES', 'MLB-BABY_PACIFIERS', 'MLB-BABY_FEEDING_PACIFIERS', 'MLB-BABIES_FORMULAS','MLB-BABY_BOTTLE_NIPPLES']))]


        self.data = total