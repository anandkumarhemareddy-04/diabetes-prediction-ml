from flask import Flask, render_template, request
import numpy as np
import pickle

# Load models
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('imputer.pkl', 'rb') as f:
    fill_zeros = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        field_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin',
                       'bmi', 'diab_pred', 'age', 'skin']

        # Get input
        form_data = {field: request.form[field] for field in field_names}

        # ✅ Input validation
        if any(float(v) <= 0 for v in form_data.values()):
            return render_template(
                'index.html',
                prediction_text="Invalid input! Values must be positive.",
                form_data=form_data
            )

        # Convert to array
        features = [float(form_data[field]) for field in field_names]
        input_array = np.asarray(features).reshape(1, -1)

        # Preprocessing
        input_array = fill_zeros.transform(input_array)
        input_array = scaler.transform(input_array)

        # Prediction
        prediction = rf_model.predict(input_array)
        prob = rf_model.predict_proba(input_array)[0][1]

        # ✅ Risk level
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.7:
            risk = "Medium"
        else:
            risk = "High"

        # ✅ Explanation
        if float(form_data['glucose_conc']) > 150:
            reason = "High Glucose"
        elif float(form_data['bmi']) > 30:
            reason = "High BMI"
        else:
            reason = "Normal Parameters"

        # Final result
        result = f"{risk} Risk ({round(prob*100,2)}%) - Reason: {reason}"

        return render_template(
            'index.html',
            prediction_text=result,
            form_data=form_data
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}',
            form_data=None
        )


if __name__ == '__main__':
    app.run(debug=True)