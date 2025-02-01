import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the data
merged_all = pd.read_csv('../DA_data/processed/merged_final.csv')

# Section 1: Calculate overall MAE for a model
def calculate_overall_mae(df, actual_col, prediction_col):
    """
    Calculate the Mean Absolute Error (MAE) between actual and predicted values.
    """
    mae = mean_absolute_error(df[actual_col], df[prediction_col])
    return mae

# Assuming merged_all contains the correct columns for actual sales and predictions
overall_mae_model_a = calculate_overall_mae(merged_all, 'actual_sales_pcs', 'sales_prediction_pcs_model_a')
overall_mae_model_b = calculate_overall_mae(merged_all, 'actual_sales_pcs', 'sales_prediction_pcs_model_b')

# Display the overall MAE for both models for comparison
print(f"Overall MAE for Model A: {overall_mae_model_a}")
print(f"Overall MAE for Model B: {overall_mae_model_b}")

# Section 2: Evaluate MAE by Market
def calculate_mae_by_market(df, markets):
    """
    Calculate and return the MAE for Model A and Model B in each market.
    """
    results = []
    for market in markets:
        market_data = df[df['market'] == market]
        mae_a = calculate_overall_mae(market_data, 'actual_sales_pcs', 'sales_prediction_pcs_model_a')
        mae_b = calculate_overall_mae(market_data, 'actual_sales_pcs', 'sales_prediction_pcs_model_b')
        results.append({'market': market, 'MAE_Model_A': mae_a, 'MAE_Model_B': mae_b})

    return pd.DataFrame(results)

markets = merged_all['market'].unique()
mae_results_by_market = calculate_mae_by_market(merged_all, markets)
print(mae_results_by_market)

# Section 3: Calculate Monthly MAE
def calculate_monthly_mae(df, actual_col, pred_col_a, pred_col_b):
    """
    Calculate and return the monthly MAE for both models.
    """
    df['date'] = df['week_nb'].apply(lambda x: datetime.strptime(f"{str(x)}-1", "%Y%W-%w"))
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['year_month'] = df['date'].dt.to_period('M')

    monthly_results = []
    for year_month in sorted(df['year_month'].unique()):
        monthly_data = df[df['year_month'] == year_month]
        mae_a = mean_absolute_error(monthly_data[actual_col], monthly_data[pred_col_a])
        mae_b = mean_absolute_error(monthly_data[actual_col], monthly_data[pred_col_b])
        monthly_results.append({'Year_Month': year_month, 'MAE_Model_A': mae_a, 'MAE_Model_B': mae_b})

    return pd.DataFrame(monthly_results)

monthly_mae_comparison = calculate_monthly_mae(merged_all, 'actual_sales_pcs', 'sales_prediction_pcs_model_a', 'sales_prediction_pcs_model_b')
print(monthly_mae_comparison)

# Section 4: Calculate MAE for Each Department and Section for Both Models
def calculate_mae_for_both_models(df, actual_col, model_a_pred_col, model_b_pred_col):
    """
    Calculate MAE for each department and section for both Model A and Model B.
    """
    df['absolute_error_model_a'] = abs(df[actual_col] - df[model_a_pred_col])
    df['absolute_error_model_b'] = abs(df[actual_col] - df[model_b_pred_col])

    mae_results = df.groupby(['department_name', 'section_name']).agg(
        MAE_Model_A=('absolute_error_model_a', 'mean'),
        MAE_Model_B=('absolute_error_model_b', 'mean')
    ).reset_index()

    return mae_results

mae_results = calculate_mae_for_both_models(merged_all, 'actual_sales_pcs', 'sales_prediction_pcs_model_a', 'sales_prediction_pcs_model_b')
print(mae_results)



# Visualize Overall MAE
def plot_overall_mae(overall_mae_model_a, overall_mae_model_b):
    """
    Plots the overall MAE for both models for comparison.
    """
    plt.figure(figsize=(8, 6))
    models = ['Model A', 'Model B']
    values = [overall_mae_model_a, overall_mae_model_b]
    plt.bar(models, values, color=['skyblue', 'orange'])
    plt.title('Overall MAE Comparison')
    plt.ylabel('MAE')
    for i, value in enumerate(values):
        plt.text(i, value + 0.5, f"{value:.2f}", ha='center')
    plt.show()

# Visualize MAE by Market
def plot_mae_by_market(mae_results_by_market):
    """
    Plots the MAE for each market for both models.
    """
    plt.figure(figsize=(14, 6))
    x = range(len(mae_results_by_market))
    plt.bar(x, mae_results_by_market['MAE_Model_A'], width=0.4, label='Model A', align='center')
    plt.bar(x, mae_results_by_market['MAE_Model_B'], width=0.4, label='Model B', align='edge')
    plt.xlabel('Market')
    plt.ylabel('MAE')
    plt.xticks(x, mae_results_by_market['market'])
    plt.title('MAE Comparison by Market')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize Monthly MAE
def plot_monthly_mae(monthly_mae_comparison):
    """
    Plots the monthly MAE for both models, with a conversion of Period objects to strings.
    """
    plt.figure(figsize=(14, 6))
    # Convert 'Year_Month' Periods to strings for plotting
    year_month_str = monthly_mae_comparison['Year_Month'].astype(str)
    plt.plot(year_month_str, monthly_mae_comparison['MAE_Model_A'], label='Model A', marker='o')
    plt.plot(year_month_str, monthly_mae_comparison['MAE_Model_B'], label='Model B', marker='x')
    plt.xlabel('Month')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.title('Monthly MAE Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Visualize MAE for Each Department and Section
def plot_mae_for_departments_and_sections(mae_results):
    """
    Plots the MAE for each department and section for both models.
    """
    plt.figure(figsize=(14, 6))
    sections = mae_results['department_name'] + ' ' + mae_results['section_name']
    x = range(len(sections))
    plt.bar(x, mae_results['MAE_Model_A'], width=0.4, label='Model A', align='center')
    plt.bar(x, mae_results['MAE_Model_B'], width=0.4, label='Model B', align='edge')
    plt.xlabel('Department and Section')
    plt.ylabel('MAE')
    plt.xticks(x, sections, rotation=90)
    plt.title('MAE Comparison by Department and Section')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Execute the plotting functions with your calculated values
plot_overall_mae(overall_mae_model_a, overall_mae_model_b)
plot_mae_by_market(mae_results_by_market)
plot_monthly_mae(monthly_mae_comparison)
plot_mae_for_departments_and_sections(mae_results)

