from flask import Flask, render_template, request
import numpy as np
import pickle

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

        form_data = {field: request.form[field] for field in field_names}
        features = [float(form_data[field]) for field in field_names]
        input_array = np.asarray(features).reshape(1, -1)

        input_array = fill_zeros.transform(input_array)
        input_array = scaler.transform(input_array)

        prediction = rf_model.predict(input_array)

        # ✅ Return exactly "Diabetic" or "Healthy" — nothing else
        result = "Diabetic" if prediction[0] == 1 else "Healthy"

        return render_template('index.html', prediction_text=result, form_data=form_data)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', form_data=None)

if __name__ == '__main__':
    app.run(debug=True)