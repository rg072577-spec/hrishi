from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('salary_model_knn.pkl')
scaler = joblib.load('scaler.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    experience = float(request.form['experience'])
    education = request.form['education']
    job = request.form['job']
    gender = request.form['gender']

    input_dict = {
        'Age': [age],
        'Years of Experience': [experience],
        'Education Level_' + education: [1],
        'Job Title_' + job: [1],
        'Gender_encoded': [gender_encoder.transform([gender])[0]]
    }

    input_df = pd.DataFrame(input_dict)
    input_df[['Age', 'Years of Experience']] = scaler.transform(input_df[['Age', 'Years of Experience']])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    predicted_salary = model.predict(input_df)[0]
    return render_template('index.html', prediction_text=f'Predicted Salary: ₹{int(predicted_salary)}')

if __name__ == '__main__':
    app.run(debug=True)