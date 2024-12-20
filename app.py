from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open('Diabetes.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map form data to variables matching the input names in the form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['bloodpressure'])
        skin_thickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetespedigreefunction'])
        age = float(request.form['age'])

        # Create a DataFrame from the input values
        inputs = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
        row_df = pd.DataFrame(inputs)

        # Scale the input data
        row_df_scaled = scaler.transform(row_df)

        # Predict the probability of diabetes (Outcome = 1)
        prediction = model.predict_proba(row_df_scaled)
        output = prediction[0][1]  # Probability of diabetes

        # Output result based on threshold (0.5)
        if output > 0.5:
            result = 'Your chance of having diabetes is high."Control Your Sugars!!'
        else:
            result = f'You are safe. Probability of having diabetes is {output:.2f}..Lets Always Lie On this side..'

        return render_template('index.html', pred=result)

    except Exception as e:
        return render_template('index.html', pred=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
