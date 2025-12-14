# mds-rely-llm-coding

## Data Analysis and Model Fitting App

A Streamlit web application for data analysis and linear regression modeling.

### Features

- **File Upload**: Upload CSV files containing your data
- **Data Visualization**: Interactive scatter plots of your data
- **Model Fitting**: Automatic linear regression model fitting
- **Model Evaluation**: Display R² score, MSE, RMSE, and model equation
- **Residual Analysis**: Visualize residuals to assess model fit

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload a CSV file with at least 2 numeric columns

4. Select the X (independent) and Y (dependent) variables

5. View the data visualization, model fitting results, and residual plots

### Sample Data

A sample CSV file (`sample_data.csv`) is included for testing the application.