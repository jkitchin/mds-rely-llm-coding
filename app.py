import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.title("Data Analysis and Model Fitting App")

st.markdown("""
This app allows you to:
- Upload a CSV file with data
- Visualize the data
- Fit a linear regression model to the data
""")

# File upload widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.write(df.head())
        
        st.subheader("Data Statistics")
        st.write(df.describe())
        
        # Check if data has at least 2 columns
        if len(df.columns) >= 2:
            # Let user select X and Y columns
            st.subheader("Select Variables for Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X variable (independent)", numeric_cols, index=0)
                y_col = st.selectbox("Select Y variable (dependent)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                
                if x_col != y_col:
                    # Extract X and Y data
                    X = df[x_col].values.reshape(-1, 1)
                    y = df[y_col].values
                    
                    # Plot the data
                    st.subheader("Data Visualization")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(X, y, alpha=0.5, label='Data points')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'{y_col} vs {x_col}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Fit linear regression model
                    st.subheader("Model Fitting")
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Make predictions
                    y_pred = model.predict(X)
                    
                    # Calculate metrics
                    r2 = r2_score(y, y_pred)
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)
                    
                    # Display model information
                    st.write(f"**Model Equation:** y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}")
                    st.write(f"**R² Score:** {r2:.4f}")
                    st.write(f"**Mean Squared Error:** {mse:.4f}")
                    st.write(f"**Root Mean Squared Error:** {rmse:.4f}")
                    
                    # Plot with fitted line
                    st.subheader("Data with Fitted Model")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.scatter(X, y, alpha=0.5, label='Data points')
                    
                    # Sort X and y_pred for proper line plotting
                    sort_idx = np.argsort(X.flatten())
                    X_sorted = X[sort_idx]
                    y_pred_sorted = y_pred[sort_idx]
                    
                    ax2.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Fitted line')
                    ax2.set_xlabel(x_col)
                    ax2.set_ylabel(y_col)
                    ax2.set_title(f'Linear Regression: {y_col} vs {x_col}')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    
                    # Residual plot
                    st.subheader("Residual Plot")
                    residuals = y - y_pred
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    ax3.scatter(y_pred, residuals, alpha=0.5)
                    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
                    ax3.set_xlabel('Predicted values')
                    ax3.set_ylabel('Residuals')
                    ax3.set_title('Residual Plot')
                    ax3.grid(True, alpha=0.3)
                    st.pyplot(fig3)
                else:
                    st.warning("Please select different columns for X and Y variables.")
            else:
                st.error("The dataset needs at least 2 numeric columns for analysis.")
        else:
            st.error("The dataset needs at least 2 columns for analysis.")
            
    except Exception as e:
        st.error(f"Error loading or processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to get started.")
    
    # Show sample data format
    st.subheader("Sample Data Format")
    st.write("Your CSV file should have at least 2 numeric columns. Example:")
    sample_data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2.1, 3.9, 6.2, 7.8, 10.1]
    })
    st.write(sample_data)
