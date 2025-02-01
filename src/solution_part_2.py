import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import calendar

# Load the data
merged_all = pd.read_csv('../DA_data/processed/merged_final.csv')
merged_all.replace([np.inf, -np.inf], np.nan, inplace=True)
# Calculate the difference between the order proposal and the actual sales of models A and B
merged_all['diff_a'] = abs(merged_all['order_suggestion_pcs_a'] - merged_all['accumulated_sales_a'])
merged_all['diff_b'] = abs(merged_all['order_suggestion_pcs_b'] - merged_all['accumulated_sales_b'])

# Calculate and compare the average difference between the two models
mean_diff_a = merged_all['diff_a'].mean()
mean_diff_b = merged_all['diff_b'].mean()

print("Model A average difference:", mean_diff_a)
print("Model B average difference:", mean_diff_b)

def plot_mean_difference(mean_diff_a, mean_diff_b):
    """
    Plots the overall average difference for both models for comparison.
    """
    plt.figure(figsize=(8, 6))
    models = ['Model A', 'Model B']
    values = [mean_diff_a, mean_diff_b]
    plt.bar(models, values, color=['skyblue', 'orange'])
    plt.title('Overall Average Difference')
    plt.ylabel('Average Difference Account')
    for i, value in enumerate(values):
        plt.text(i, value + 0.5, f"{value:.2f}", ha='center')
    plt.show()

plot_mean_difference(mean_diff_a, mean_diff_b)

# Set histogram parameters
plt.figure(figsize=(12, 8))
bin_width = (max(merged_all['diff_a'].max(), merged_all['diff_b'].max()) - 0) / 40  # Defines the width of a single bin
bins = np.arange(0, max(merged_all['diff_a'].max(), merged_all['diff_b'].max()) + bin_width, bin_width)  # Define bin boundaries

# Calculate the center location of each bin
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Histogram statistics were performed on the difference data of each model
hist_a, _ = np.histogram(merged_all['diff_a'], bins)
hist_b, _ = np.histogram(merged_all['diff_b'], bins)

# Sets the offset between bar charts
offset = bin_width * 0.25

# Draw A histogram for model A
plt.bar(bin_centers - offset, hist_a, width=offset * 2, label='Model A', color='blue', alpha=0.5)

# Draw a histogram for model B
plt.bar(bin_centers + offset, hist_b, width=offset * 2, label='Model B', color='red', alpha=0.5)

plt.title('Absolute Differences between Order Suggestions and Actual Sales')
plt.xlabel('Absolute Difference')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

"""Analysis Section 2"""
# Load the data
corrected_merged_stock = merged_all.copy()

# Define a function to convert year-week to year, month, and quarter
def week_to_year_month_quarter(year_week):
    year, week = divmod(year_week, 100)
    first_day_of_year = datetime(year, 1, 1)
    week_start = first_day_of_year + timedelta(weeks=week-1, days=-first_day_of_year.weekday())
    month = week_start.month
    quarter = (month - 1) // 3 + 1
    return year % 100, month, quarter

# Apply the function to each row to extract year, month, and quarter
corrected_merged_stock['year'], corrected_merged_stock['month'], corrected_merged_stock['quarter'] = zip(
    *corrected_merged_stock['week_nb'].apply(week_to_year_month_quarter)
)

# Group and calculate the average end-of-season stock for each year-month and year-quarter for Model A and B
monthly_comparison = corrected_merged_stock.groupby(['year', 'month']).agg(
    avg_stock_a=('stock_a', 'mean'),
    avg_stock_b=('stock_b', 'mean')
).reset_index()

quarterly_comparison = corrected_merged_stock.groupby(['year', 'quarter']).agg(
    avg_stock_a=('stock_a', 'mean'),
    avg_stock_b=('stock_b', 'mean')
).reset_index()

# Print comparisons
print("Monthly Comparison of Average End-of-Season Stock for Model A and Model B:")
print(monthly_comparison)
print("\nQuarterly Comparison of Average End-of-Season Stock for Model A and Model B:")
print(quarterly_comparison)

# Combine year and month, and year and quarter for plotting
monthly_comparison['year_month'] = monthly_comparison.apply(lambda x: f"{int(x['year'])}-{calendar.month_abbr[int(x['month'])]}", axis=1)
quarterly_comparison['year_quarter'] = quarterly_comparison.apply(lambda x: f"{int(x['year'])}-Q{int(x['quarter'])}", axis=1)

# Plotting the monthly comparison with the adjusted year_month format
plt.figure(figsize=(15, 7))
sns.lineplot(data=monthly_comparison, x='year_month', y='avg_stock_a', marker='o', label='Model A')
sns.lineplot(data=monthly_comparison, x='year_month', y='avg_stock_b', marker='o', label='Model B', color='red')
plt.xticks(rotation=45)
plt.title('Monthly Comparison of Average End-of-Season Stock for Model A and Model B')
plt.xlabel('Year-Month')
plt.ylabel('Average Stock')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the quarterly comparison directly as the format already matches the desired format
plt.figure(figsize=(15, 7))
sns.lineplot(data=quarterly_comparison, x='year_quarter', y='avg_stock_a', marker='o', label='Model A')
sns.lineplot(data=quarterly_comparison, x='year_quarter', y='avg_stock_b', marker='o', label='Model B', color='red')
plt.xticks(rotation=45)
plt.title('Quarterly Comparison of Average End-of-Season Stock for Model A and Model B')
plt.xlabel('Year-Quarter')
plt.ylabel('Average Stock')
plt.legend()
plt.tight_layout()
plt.show()

"""Analysis section 3"""
# Load the data
merged_stock_data = merged_all.copy()

# Assuming a holding cost rate of 5% annually, converting to weekly
annual_holding_cost_rate = 0.05
weeks_in_year = 52
weekly_holding_cost_rate = annual_holding_cost_rate / weeks_in_year

# Calculate the weekly profit for each model
merged_stock_data['profit_a'] = (merged_stock_data['actual_sales_pcs'] * merged_stock_data['price_sek']) - \
                                (merged_stock_data['order_suggestion_pcs_a'] * merged_stock_data['cost_sek']) - \
                                (merged_stock_data['stock_a'] * merged_stock_data['cost_sek'] * weekly_holding_cost_rate)
merged_stock_data['profit_b'] = (merged_stock_data['actual_sales_pcs'] * merged_stock_data['price_sek']) - \
                                (merged_stock_data['order_suggestion_pcs_b'] * merged_stock_data['cost_sek']) - \
                                (merged_stock_data['stock_b'] * merged_stock_data['cost_sek'] * weekly_holding_cost_rate)

# Calculate cumulative profit for each model
merged_stock_data['cumulative_profit_a'] = merged_stock_data['profit_a'].cumsum()
merged_stock_data['cumulative_profit_b'] = merged_stock_data['profit_b'].cumsum()

# Define functions to convert week number to month and quarter in the desired format
# Function to convert week number to year and month

def week_to_quarter(year_week):
    year, week = divmod(year_week, 100)
    first_day_of_year = pd.Timestamp(year=year, month=1, day=1)
    day_of_week = first_day_of_year + pd.Timedelta(weeks=week-1)
    quarter = (day_of_week.month - 1) // 3 + 1
    return f"{year % 100}-Q{quarter}"

# Apply functions to get the month and quarter
merged_stock_data['quarter'] = merged_stock_data['week_nb'].apply(week_to_quarter)

# Group by month and quarter for the cumulative profit of each model

quarterly_data = merged_stock_data.groupby('quarter').agg(
    cumulative_profit_a=('cumulative_profit_a', 'last'),
    cumulative_profit_b=('cumulative_profit_b', 'last')
).reset_index()


# Plot the quarterly cumulative profit
plt.figure(figsize=(14, 7))
plt.plot(quarterly_data['quarter'], quarterly_data['cumulative_profit_a'], label='Model A', marker='o')
plt.plot(quarterly_data['quarter'], quarterly_data['cumulative_profit_b'], label='Model B', marker='x')
plt.title('Quarterly Cumulative Profit Comparison')
plt.xlabel('Quarter')
plt.ylabel('Cumulative Profit')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# First, we will create a 'date' column in our DataFrame which contains the dates for each week number
def convert_week_to_date(year_week):
    year, week = divmod(year_week, 100)
    # Compute the Monday of the given week number
    first_day = pd.Timestamp(year=year, month=1, day=1)
    if first_day.weekday() != 0:
        first_day = first_day + pd.Timedelta(days=(7-first_day.weekday()))
    return first_day + pd.Timedelta(weeks=(week-1))

# Apply the conversion to create the 'date' column
merged_stock_data['date'] = merged_stock_data['week_nb'].apply(convert_week_to_date)

# Group by month and quarter using the actual dates
merged_stock_data['month'] = merged_stock_data['date'].dt.strftime('%y-%b')
# Get the last entry for each month and quarter
monthly_cumulative_profit = merged_stock_data.groupby('month', sort=False).agg({
    'cumulative_profit_a': 'last',
    'cumulative_profit_b': 'last'
}).reset_index()

# Correct order of months by sorting by date
monthly_cumulative_profit['month'] = pd.to_datetime(monthly_cumulative_profit['month'], format='%y-%b')
monthly_cumulative_profit.sort_values('month', inplace=True)
monthly_cumulative_profit['month'] = monthly_cumulative_profit['month'].dt.strftime('%y-%b')

# Plot the monthly cumulative profit
plt.figure(figsize=(15, 7))
plt.plot(monthly_cumulative_profit['month'], monthly_cumulative_profit['cumulative_profit_a'], label='Model A', marker='o')
plt.plot(monthly_cumulative_profit['month'], monthly_cumulative_profit['cumulative_profit_b'], label='Model B', marker='x')
plt.title('Monthly Cumulative Profit Comparison')
plt.xlabel('Month')
plt.ylabel('Cumulative Profit')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()