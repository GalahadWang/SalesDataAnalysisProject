# Sales Data Analysis Project

## Overview

This project is structured around the analysis of sales data to evaluate and compare the performance of two machine learning models (Model A and Model B) in predicting future sales. The project is divided into four main parts: data preparation, RMSE analysis, MAE analysis, and further insights into stock and profit analysis.

## Project Structure

### 1. Data Preparation (`prepare_enriched_sales_data.py`)

**Objective:** Prepare and enrich sales data by merging various datasets, including actual sales, article hierarchy, model predictions, and price and cost data.

**Key Steps:**

- Load and merge CSV files into a single enriched DataFrame.
- Calculate accumulated sales and prepare stock calculations for both models.
- Save the enriched DataFrame for further analysis.

### 2. RMSE Analysis (`solution_part1(RMSE).py`)

**Objective:** Analyze the prediction accuracy of both models using the Root Mean Square Error (RMSE) metric.

**Key Analyses:**

- Calculate overall RMSE for both models.
- Perform market-wise RMSE analysis.
- Analyze monthly RMSE and by department and section.
- Visualizations include RMSE comparisons by market, month, and department/section.

### 3. MAE Analysis (`solution_part1(MAE).py`)

**Objective:** Complement the RMSE analysis by evaluating the models' accuracy using the Mean Absolute Error (MAE) metric.

**Key Analyses:**

- Calculate overall MAE for both models.
- Perform market-wise MAE analysis.
- Analyze monthly MAE and by department and section.
- Visualizations include MAE comparisons by market, month, and department/section.

### 4. Further Insights (`solution_part2.py`)

**Objective:** Dive deeper into the analysis of model performance, focusing on stock and profit evaluation.

**Key Analyses:**

- Compare the average differences between order suggestions and actual sales.
- Analyze average end-of-season stock monthly and quarterly.
- Calculate and compare the cumulative profit for both models quarterly and monthly.
- Visualizations include histograms of differences, stock level comparisons, and cumulative profit trends.

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

bashCopy code

`pip install pandas numpy matplotlib seaborn scikit-learn`

### Running the Scripts

Execute the scripts in the following order to perform the complete analysis:

1. `python src/prepare_enriched_sales_data.py`
2. `python src/solution_part1(RMSE).py`
3. `python src/solution_part1(MAE).py`
4. `python src/solution_part_2.py`

## Visualization Highlights

- **RMSE and MAE Comparisons:** Insights into the accuracy of models A and B.
- **Stock Level Analysis:** Evaluation of model performance in inventory management.
- **Profit Analysis:** Financial performance comparison based on weekly profit and cumulative profit calculations.

## Contributors

- Jieda(Jay) Wang
- Email: Jiedaawang@gmail.com

## Acknowledgments

- This project utilizes publicly available or anonymized data originally derived from retail sales analysis.
- The project is solely for **technical demonstration and portfolio purposes** and is **not affiliated with, endorsed by, or officially associated with any company, including H&M**.

### Disclaimer
This repository is for educational and portfolio purposes only. The data used in this project has been anonymized and does not represent any real-world company or commercial use case. No proprietary or confidential information is included.

