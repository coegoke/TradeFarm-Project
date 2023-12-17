from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Model and scaler dictionary
model_scaler_dict = {
    ('Jawa Timur', 'Minyak Goreng'): ('model_Jawa Timur_Minyak Goreng.h5', 'Minyak Goreng'),
    ('Jawa Timur', 'Beras'): ('model_Jawa Timur_Beras.h5', 'Beras'),
    ('Jawa Timur', 'Bawang Merah'): ('model_Jawa Timur_Bawang Merah.h5', 'Bawang Merah'),
    ('Jawa Timur', 'Cabai Merah'): ('model_Jawa Timur_Cabai Merah.h5', 'Cabai Merah'),
    ('Jawa Timur', 'Cabai Rawit'): ('model_Jawa Timur_Cabai Rawit.h5', 'Cabai Rawit'),
    ('Jawa Timur', 'Gula Pasir'): ('model_Jawa Timur_Gula Pasir.h5', 'Gula Pasir'),
    ('Jawa Tengah', 'Minyak Goreng'): ('model_Jawa Tengah_Minyak Goreng.h5', 'Minyak Goreng'),
    ('Jawa Tengah', 'Beras'): ('model_Jawa Tengah_Beras.h5', 'Beras'),
    ('Jawa Tengah', 'Bawang Merah'): ('model_Jawa Tengah_Bawang Merah.h5', 'Bawang Merah'),
    ('Jawa Tengah', 'Cabai Merah'): ('model_Jawa Tengah_Cabai Merah.h5', 'Cabai Merah'),
    ('Jawa Tengah', 'Cabai Rawit'): ('model_Jawa Tengah_Cabai Rawit.h5', 'Cabai Rawit'),
    ('Jawa Tengah', 'Gula Pasir'): ('model_Jawa Tengah_Gula Pasir.h5', 'Gula Pasir'),
    ('Jawa Barat', 'Minyak Goreng'): ('model_Jawa Barat_Minyak Goreng.h5', 'Minyak Goreng'),
    ('Jawa Barat', 'Beras'): ('model_Jawa Barat_Beras.h5', 'Beras'),
    ('Jawa Barat', 'Bawang Merah'): ('model_Jawa Barat_Bawang Merah.h5', 'Bawang Merah'),
    ('Jawa Barat', 'Cabai Merah'): ('model_Jawa Barat_Cabai Merah.h5', 'Cabai Merah'),
    ('Jawa Barat', 'Cabai Rawit'): ('model_Jawa Barat_Cabai Rawit.h5', 'Cabai Rawit'),
    ('Jawa Barat', 'Gula Pasir'): ('model_Jawa Barat_Gula Pasir.h5', 'Gula Pasir'),
    ('Banten', 'Minyak Goreng'): ('model_Banten_Minyak Goreng.h5', 'Minyak Goreng'),
    ('Banten', 'Beras'): ('model_Banten_Beras.h5', 'Beras'),
    ('Banten', 'Bawang Merah'): ('model_Banten_Bawang Merah.h5', 'Bawang Merah'),
    ('Banten', 'Cabai Merah'): ('model_Banten_Cabai Merah.h5', 'Cabai Merah'),
    ('Banten', 'Cabai Rawit'): ('model_Banten_Cabai Rawit.h5', 'Cabai Rawit'),
    ('Banten', 'Gula Pasir'): ('model_Banten_Gula Pasir.h5', 'Gula Pasir'),
}

# Load data once
data_jawa_timur = pd.read_csv('data/jawa timur.csv')
data_jawa_barat = pd.read_csv('data/jawa barat.csv')
data_banten = pd.read_csv('data/banten.csv')
data_jawa_tengah = pd.read_csv('data/Jawa Tengah.csv')

# Function to perform future forecasting
def perform_forecasting(model, series, target_date):
    
    future_forecast = []

    # Calculate the number of days to predict
    days_to_predict = (target_date - pd.to_datetime("2023-11-30")).days
    input_data = series[(days_to_predict-30):days_to_predict][np.newaxis]
    for _ in range(30):
        prediction = model.predict(input_data)
        future_forecast.append(prediction[0, 0])
        input_data = np.append(input_data[:, 1:, :], [[prediction[0, 0]]], axis=1)
    
    return future_forecast,days_to_predict

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    target_date_str = request.json['tanggal']
    target_date = pd.to_datetime(target_date_str, format="%Y-%m-%d")
    nama_daerah = request.json['daerah']
    nama_makanan = request.json['makanan']

    # Check if the combination exists in the dictionary
    model_scaler_key = (nama_daerah, nama_makanan)
    if model_scaler_key not in model_scaler_dict:
        return jsonify({'error': 'Model not found for the given region and food.'})
    
    model_filename, makanan_label = model_scaler_dict[model_scaler_key]

    if nama_daerah == 'Jawa Timur':
        data = data_jawa_timur
    elif nama_daerah == 'Jawa Barat':
        data = data_jawa_barat
    elif nama_daerah == 'Jawa Tengah':
        data = data_jawa_tengah
    elif nama_daerah == 'Banten':
        data = data_banten
    
    selected_model = tf.keras.models.load_model(f'model/{model_filename}')
    
    # Perform future forecasting until the target date
    price_data = np.array(data[makanan_label]).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = scaler.fit_transform(price_data.reshape(-1, 1))
    future_forecast,days = perform_forecasting(selected_model, series, target_date)

    # Convert the result to a list before sending it as JSON
    result = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten().tolist()
    future_dates = pd.date_range(start=data['Tanggal'].iloc[-1], periods=days+1)[1:]
    future_dates_str = future_dates[-30:].astype(str).tolist()

    return jsonify({'date forecast': future_dates_str, 'future_forecast': result})


if __name__ == '__main__':
    app.run(debug=True)
