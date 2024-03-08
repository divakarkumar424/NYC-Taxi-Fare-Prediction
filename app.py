# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Load the pre-trained model
model = pickle.load(open('NYC_fare.pkl', 'rb'))

# Create a Flask web app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        input_data = {
            'pickup_longitude': float(request.form['pickup_longitude']),
            'pickup_latitude': float(request.form['pickup_latitude']),
            'dropoff_longitude': float(request.form['dropoff_longitude']),
            'dropoff_latitude': float(request.form['dropoff_latitude']),
            'passenger_count': int(request.form['passenger_count']),
        }

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Make the prediction
        prediction = model.predict(input_df)[0]

        # Display the predicted fare
        return render_template('index.html', prediction_text=f'Predicted Fare: ${prediction:.2f}')

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('index.html', prediction_text='Error occurred. Please check your input.')

if __name__ == '__main__':
    app.run(debug=True)

