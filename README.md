# mds-rely-llm-coding

A simple web application to upload CSV files, visualize data, and fit a linear regression model.

## Features

- Upload CSV files through a web interface
- Automatic detection of numeric columns
- Data visualization with scatter plot
- Linear regression model fitting
- Display of model parameters (slope, intercept, R² score)
- Data preview of uploaded CSV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jkitchin/mds-rely-llm-coding.git
cd mds-rely-llm-coding
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a CSV file:
   - Click "Choose CSV File" button
   - Select a CSV file with at least 2 numeric columns
   - Click "Upload and Analyze"

4. View the results:
   - Scatter plot with fitted linear regression line
   - Model parameters (equation, R² score)
   - Preview of the uploaded data

## Requirements

- Python 3.7+
- Flask
- pandas
- matplotlib
- numpy
- scikit-learn

## CSV File Format

The CSV file should:
- Have at least 2 numeric columns
- Use standard CSV format (comma-separated)
- The first two numeric columns will be used for X and Y axes

Example CSV:
```csv
x,y
1,2.1
2,4.2
3,5.9
4,8.1
5,10.2
```