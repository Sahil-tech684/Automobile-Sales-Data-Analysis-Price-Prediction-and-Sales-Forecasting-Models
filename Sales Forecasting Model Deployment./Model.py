from flask import Flask, render_template, request
import pickle
# import os

# SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
# MODEL_PATH = os.path.join(SCRIPT_PATH, 'Sales Forecasting Model Deployment', 'Model.pkl')

app = Flask(__name__)

sarima_results = None

try:
    with open('Sales Forecasting Model Deployment/Model.pkl', 'rb') as file:
        sarima_results = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/')
def home():
    return render_template('sales_model_page.html')

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    global sarima_results  

    if request.method == 'POST':
        num_months = int(request.form['num_months'])

        if sarima_results is not None:
            try:
                forecast_periods = num_months
                forecast = sarima_results.get_forecast(steps=forecast_periods)
                
                forecast_dates = forecast.row_labels
                forecast_values = forecast.predicted_mean.tolist()
                confidence_interval = forecast.conf_int()

                # Extract confidence interval values for each forecasted date
                confidence_interval_values = confidence_interval.values.tolist()

                return render_template('sales_result.html', num_months=num_months, sales_predictions=zip(forecast_dates, forecast_values, confidence_interval_values))
            except Exception as e:
                print(f"Error making prediction: {e}")
                return render_template('error.html', error_message="Error making prediction. Please try again.")
        else:
            print("Model is not loaded.")
            return render_template('error.html', error_message="Model not loaded. Please try again.")
    
    return "Invalid request"

if __name__ == '__main__':
    app.run(debug=True)
