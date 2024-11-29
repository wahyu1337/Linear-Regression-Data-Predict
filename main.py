import pandas as pd
from flask import Flask, request, redirect, url_for, render_template, jsonify, session
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
app.secret_key = 'regresi_linear'

# Create dataset directory if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        form = request.form
        filepath = os.path.join('dataset', secure_filename(file.filename))
        file.save(filepath)
        data = pd.read_csv(filepath)

        # Extract features and target
        Fitur = data[['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']]
        Target = data['Produksi']

        # Split data into training and testing
        Fitur_train, Fitur_test, Target_train, Target_test = train_test_split(
            Fitur, Target, test_size=float(form['data_uji']), random_state=42)

        # Train the model
        model = LinearRegression(fit_intercept=True)
        model.fit(Fitur_train, Target_train)

        # Model evaluation
        Target_pred = model.predict(Fitur_test)
        r2 = r2_score(Target_test, Target_pred)

        # Prediction input
        luas_panen = float(form['M2'])
        curah_hujan = float(form['mm'])
        kelembapan = float(form['%'])
        suhu = float(form['(C)'])
        input_data = pd.DataFrame(
            [[luas_panen, curah_hujan, kelembapan, suhu]],
            columns=['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
        )
        hasil_prediksi = model.predict(input_data)

        # Store results in session
        session['hasil_evaluasi'] = {
            'koefisien': round(r2, 3),
            'produksi': f"Prediksi produksi: {round(hasil_prediksi[0], 2)} Kg"
        }
        return redirect(url_for('evaluate'))

    return "Error: Pastikan file yang diunggah berformat CSV"

@app.route('/evaluate')
def evaluate():
    evaluation = session.get('hasil_evaluasi', {})
    return render_template('evaluasi.html', evaluation=evaluation)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
