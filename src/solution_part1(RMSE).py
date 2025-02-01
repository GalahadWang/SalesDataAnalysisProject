import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


# Load the data
merged_all = pd.read_csv('../DA_data/processed/merged_final.csv')

# Section 1
# Function to calculate overall RMSE for a model
def calculate_overall_rmse(df, actual_col, prediction_col):
    rmse = sqrt(mean_squared_error(df[actual_col], df[prediction_col]))
    return rmse

# Assuming merged_all contains the correct columns for actual sales and predictions
overall_rmse_model_a = calculate_overall_rmse(merged_all, 'actual_sales_pcs', 'sales_prediction_pcs_model_a')
overall_rmse_model_b = calculate_overall_rmse(merged_all, 'actual_sales_pcs', 'sales_prediction_pcs_model_b')

# Display the overall RMSE for both models for comparison
print(f"Overall RMSE for Model A: {overall_rmse_model_a}")
print(f"Overall RMSE for Model B: {overall_rmse_model_b}")

# Section 2
# Define a function that evaluates RMSE
def calculate_rmse(data, actual_col, prediction_col):
    return sqrt(mean_squared_error(data[actual_col], data[prediction_col]))

# The RMSE for Model A and Model B in each market is calculated separately
markets = merged_all['market'].unique()
results = []

for market in markets:
    market_data = merged_all[merged_all['market'] == market]
    rmse_a = calculate_rmse(market_data, 'actual_sales_pcs', 'sales_prediction_pcs_model_a')
    rmse_b = calculate_rmse(market_data, 'actual_sales_pcs', 'sales_prediction_pcs_model_b')
    results.append({'market': market, 'RMSE_Model_A': rmse_a, 'RMSE_Model_B': rmse_b})

# Convert the result to a DataFrame for easy analysis and visualization
results_df = pd.DataFrame(results)

# Display result
print(results_df)

# Section 3
# Extract the year and number of weeks, convert to the date (the first day of the selected week), and then extract the month
merged_all['date'] = merged_all['week_nb'].apply(lambda x: datetime.strptime(f"{str(x)}-1", "%Y%W-%w"))
merged_all['month'] = merged_all['date'].dt.month
merged_all['year'] = merged_all['date'].dt.year
merged_all['year_month'] = merged_all['date'].dt.to_period('M')

# Calculate the monthly RMSE
def calculate_monthly_rmse(df, actual_col, pred_col_a, pred_col_b):
    monthly_results = []
    for year_month in sorted(df['year_month'].unique()):
        monthly_data = df[df['year_month'] == year_month]

        rmse_a = sqrt(mean_squared_error(monthly_data[actual_col], monthly_data[pred_col_a]))
        rmse_b = sqrt(mean_squared_error(monthly_data[actual_col], monthly_data[pred_col_b]))

        monthly_results.append({'Year_Month': year_month, 'RMSE_Model_A': rmse_a, 'RMSE_Model_B': rmse_b})

    return pd.DataFrame(monthly_results)

monthly_rmse_comparison = calculate_monthly_rmse(merged_all, 'actual_sales_pcs', 'sales_prediction_pcs_model_a',
                                                 'sales_prediction_pcs_model_b')

print(monthly_rmse_comparison)

# Section 4
# Function to calculate RMSE for each department and section for both Model A and Model B
def calculate_rmse_for_both_models(df, actual_col, model_a_pred_col, model_b_pred_col):
    # Calculate squared error for both models
    df['squared_error_model_a'] = (df[actual_col] - df[model_a_pred_col]) ** 2
    df['squared_error_model_b'] = (df[actual_col] - df[model_b_pred_col]) ** 2

    # Group by department and section, then calculate mean squared error and take the square root for both models
    rmse_results = df.groupby(['department_name', 'section_name']).agg(
        RMSE_Model_A=('squared_error_model_a', lambda x: sqrt(x.mean())),
        RMSE_Model_B=('squared_error_model_b', lambda x: sqrt(x.mean()))
    ).reset_index()

    return rmse_results


# Assuming merged_all contains the correct columns for actual sales and predictions
rmse_results = calculate_rmse_for_both_models(merged_all, 'actual_sales_pcs', 'sales_prediction_pcs_model_a',
                                              'sales_prediction_pcs_model_b')

# Display the results for comparison
print(rmse_results)


# Visualize Overall RMSE
def plot_overall_rmse(overall_rmse_model_a, overall_rmse_model_b):
    """
    Plots the overall RMSE for both models for comparison.
    """
    plt.figure(figsize=(8, 6))
    models = ['Model A', 'Model B']
    values = [overall_rmse_model_a, overall_rmse_model_b]
    plt.bar(models, values, color=['skyblue', 'orange'])
    plt.title('Overall RMSE Comparison')
    plt.ylabel('RMSE')
    for i, value in enumerate(values):
        plt.text(i, value + 0.5, f"{value:.2f}", ha='center')
    plt.show()

# Visualize RMSE by Market
def plot_rmse_by_market(results_df):
    """
    Plots the RMSE for each market for both models.
    """
    plt.figure(figsize=(14, 6))
    x = range(len(results_df))
    plt.bar(x, results_df['RMSE_Model_A'], width=0.4, label='Model A', align='center')
    plt.bar(x, results_df['RMSE_Model_B'], width=0.4, label='Model B', align='edge')
    plt.xlabel('Market')
    plt.ylabel('RMSE')
    plt.xticks(x, results_df['market'])
    plt.title('RMSE Comparison by Market')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize Monthly RMSE
def plot_monthly_rmse(monthly_rmse_comparison):
    """
    Plots the monthly RMSE for both models, with a conversion of Period objects to strings.
    """
    plt.figure(figsize=(14, 6))
    # Convert 'Year_Month' Periods to strings for plotting
    year_month_str = monthly_rmse_comparison['Year_Month'].astype(str)
    plt.plot(year_month_str, monthly_rmse_comparison['RMSE_Model_A'], label='Model A', marker='o')
    plt.plot(year_month_str, monthly_rmse_comparison['RMSE_Model_B'], label='Model B', marker='x')
    plt.xlabel('Month')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.title('Monthly RMSE Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Visualize RMSE for Each Department and Section
def plot_rmse_for_departments_and_sections(rmse_results):
    """
    Plots the RMSE for each department and section for both models.
    """
    plt.figure(figsize=(14, 6))
    sections = rmse_results['department_name'] + ' ' + rmse_results['section_name']
    x = range(len(sections))
    plt.bar(x, rmse_results['RMSE_Model_A'], width=0.4, label='Model A', align='center')
    plt.bar(x, rmse_results['RMSE_Model_B'], width=0.4, label='Model B', align='edge')
    plt.xlabel('Department and Section')
    plt.ylabel('RMSE')
    plt.xticks(x, sections, rotation=90)
    plt.title('RMSE Comparison by Department and Section')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Execute the plotting functions with your calculated values
plot_overall_rmse(overall_rmse_model_a, overall_rmse_model_b)
plot_rmse_by_market(results_df)
plot_monthly_rmse(monthly_rmse_comparison)
plot_rmse_for_departments_and_sections(rmse_results)