import pandas as pd
import numpy as np

# Load CSV files into Pandas DataFrames
actual_sales_path = '../DA_data/raw/Actual_sales.csv'
article_hierarchy_path = '../DA_data/raw/article_hierarchy.csv'
model_a_path = '../DA_data/processed/model_A.csv'
model_b_path = '../DA_data/processed/model_B.csv'
price_and_cost_path = '../DA_data/raw/price_and_cost.csv'

actual_sales = pd.read_csv(actual_sales_path)
article_hierarchy = pd.read_csv(article_hierarchy_path)
model_a = pd.read_csv(model_a_path)
model_b = pd.read_csv(model_b_path)
price_and_cost = pd.read_csv(price_and_cost_path)

# Merge external data into the actual sales data for enrichment
actual_sales_enriched = pd.merge(actual_sales, article_hierarchy, on='article_id', how='left')
actual_sales_enriched = pd.merge(actual_sales_enriched, price_and_cost, on='article_id', how='left')

# Merge model predictions into the enriched sales data, adding specific suffixes to distinguish between models
actual_sales_enriched = pd.merge(actual_sales_enriched,
                                 model_a.rename(columns={'sales_prediction_pcs': 'sales_prediction_pcs_model_a',
                                                         'order_suggestion_pcs': 'order_suggestion_pcs_a'}),
                                 on=['article_id', 'week_nb', 'market'], how='left')
actual_sales_enriched = pd.merge(actual_sales_enriched,
                                 model_b.rename(columns={'sales_prediction_pcs': 'sales_prediction_pcs_model_b',
                                                         'order_suggestion_pcs': 'order_suggestion_pcs_b'}),
                                 on=['article_id', 'week_nb', 'market'], how='left')


# Initialize columns for accumulated sales, to be filled in the loop below
actual_sales_enriched['accumulated_sales_a'] = 0
actual_sales_enriched['accumulated_sales_b'] = 0

# Variables to keep track of the start of a new order period and accumulated sales for both models
previous_start_index_a, previous_start_index_b = 0, 0
accumulated_sales_a, accumulated_sales_b = 0, 0

# Loop through each row to calculate accumulated sales for model A and model B
for i, row in actual_sales_enriched.iterrows():
    # Handling for model A
    if row['order_suggestion_pcs_a'] > 0:
        if i > 0:
            actual_sales_enriched.at[previous_start_index_a, 'accumulated_sales_a'] = accumulated_sales_a
        accumulated_sales_a = row['actual_sales_pcs']
        previous_start_index_a = i
    else:
        accumulated_sales_a += row['actual_sales_pcs']

    # Handling for model B
    if row['order_suggestion_pcs_b'] > 0:
        if i > 0:
            actual_sales_enriched.at[previous_start_index_b, 'accumulated_sales_b'] = accumulated_sales_b
        accumulated_sales_b = row['actual_sales_pcs']
        previous_start_index_b = i
    else:
        accumulated_sales_b += row['actual_sales_pcs']

# Update the last accumulated sales period for both models
actual_sales_enriched.at[previous_start_index_a, 'accumulated_sales_a'] = accumulated_sales_a
actual_sales_enriched.at[previous_start_index_b, 'accumulated_sales_b'] = accumulated_sales_b

# Sort data to prepare for stock calculation, ensuring continuity within each article-market combination
actual_sales_enriched.sort_values(by=['article_id', 'market', 'week_nb'], inplace=True)
# Initialize stock columns
actual_sales_enriched['stock_a'] = 0
actual_sales_enriched['stock_b'] = 0

# Loop through sorted data to calculate stock based on order suggestions and actual sales
for i in range(len(actual_sales_enriched)):
    if i == 0 or (actual_sales_enriched.iloc[i]['article_id'] != actual_sales_enriched.iloc[i - 1]['article_id'] or
                  actual_sales_enriched.iloc[i]['market'] != actual_sales_enriched.iloc[i - 1]['market']):
        # Initialize stock for a new article-market combination
        actual_sales_enriched.loc[i, 'stock_a'] = actual_sales_enriched.loc[i, 'order_suggestion_pcs_a']
        actual_sales_enriched.loc[i, 'stock_b'] = actual_sales_enriched.loc[i, 'order_suggestion_pcs_b']
    else:
        # Update stock based on previous values, new orders, and sales
        actual_sales_enriched.loc[i, 'stock_a'] = (actual_sales_enriched.loc[i - 1, 'stock_a'] +
                                                   actual_sales_enriched.loc[i, 'order_suggestion_pcs_a'] -
                                                   actual_sales_enriched.loc[i, 'actual_sales_pcs'])
        actual_sales_enriched.loc[i, 'stock_b'] = (actual_sales_enriched.loc[i - 1, 'stock_b'] +
                                                   actual_sales_enriched.loc[i, 'order_suggestion_pcs_b'] -
                                                   actual_sales_enriched.loc[i, 'actual_sales_pcs'])

# The final DataFrame 'actual_sales_enriched' now contains enriched sales data with predictions, accumulated sales, and stock calculations.
# Specify the output path for the CSV file
output_csv_path = '../DA_data/processed/merged_final.csv'

# Save the DataFrame to a CSV file, without the index
actual_sales_enriched.to_csv(output_csv_path, index=False)

print(f"Data saved to {output_csv_path}")
