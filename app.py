import os
import io
import base64
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
# Note: For production, use a fixed secret key from environment variables
# e.g., app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def create_plot_and_fit(df):
    """Create a plot with data and fitted model"""
    # Try to identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None, "CSV file must contain at least 2 numeric columns for plotting and fitting."
    
    # Use first two numeric columns for x and y
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    
    # Remove any rows with missing values
    plot_df = df[[x_col, y_col]].dropna()
    
    if len(plot_df) < 2:
        return None, "Not enough data points after removing missing values."
    
    X = plot_df[x_col].values.reshape(-1, 1)
    y = plot_df[y_col].values
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6, label='Data', color='blue')
    ax.plot(X, y_pred, color='red', linewidth=2, label=f'Linear Fit (R² = {r_squared:.4f})')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} vs {x_col}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Model details
    model_info = {
        'x_column': x_col,
        'y_column': y_col,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r_squared': r_squared,
        'n_points': len(plot_df)
    }
    
    return img_base64, model_info

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Read CSV file
                df = pd.read_csv(file)
                
                if df.empty:
                    flash('CSV file is empty')
                    return redirect(request.url)
                
                # Create plot and fit model
                result = create_plot_and_fit(df)
                
                if result[0] is None:
                    flash(result[1])
                    return redirect(request.url)
                
                img_base64, model_info = result
                
                return render_template('index.html', 
                                     image=img_base64, 
                                     model_info=model_info,
                                     data_preview=df.head(10).to_html(classes='table table-striped', escape=True))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    # Note: Debug mode and host='0.0.0.0' are for development only
    # For production, set debug=False and configure proper host/port
    app.run(debug=True, host='0.0.0.0', port=5000)
