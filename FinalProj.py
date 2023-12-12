import os
import wfdb
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
import pywt
import matplotlib.pyplot as plt

data_directory = 'data'
file_extensions = ['.q1c', '.hea', '.dat']
all_labels = []
desired_length = 1000

def process_labels(labels, desired_length):
    if len(labels) < desired_length:
        processed_labels = np.pad(labels, (0, desired_length - len(labels)))
    elif len(labels) > desired_length:
        processed_labels = labels[:desired_length]
    else:
        processed_labels = labels
    return processed_labels

def process_annotation(annotation):
    samples = annotation.sample
    symbols = annotation.symbol
    labels = [1 if symbol == 'u' else 0 for symbol in symbols]
    return labels

def process_signals(signals, desired_length):
    if len(signals) < desired_length:
        processed_signals = np.pad(signals, (0, desired_length - len(signals)))
    elif len(signals) > desired_length:
        processed_signals = signals[:desired_length]
    else:
        processed_signals = signals
    return processed_signals

def apply_wavelet_transform(signals, wavelet='db1', level=1):
    wavelet_transformed_signals = np.zeros_like(signals, dtype=float)
    for i in range(signals.shape[1]):
        signal = signals[:, i]
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        wavelet_transformed_signals[:, i] = np.concatenate(coeffs)
    return wavelet_transformed_signals

all_wavelet_transformed_signals = []

for file_extension in file_extensions:
    file_list = [f for f in os.listdir(data_directory) if f.endswith(file_extension)]
    for file in file_list:
        file_path = os.path.join(data_directory, file)

        if file_extension == '.q1c':
            annotation = wfdb.rdann(file_path.replace('.q1c', ''), 'q1c')
            labels = process_annotation(annotation)
            labels = process_labels(labels, desired_length)
            all_labels.append(labels)

        elif file_extension == '.hea':
            record = wfdb.rdheader(file_path.replace('.hea', ''))

        elif file_extension == '.dat':
            signals, fields = wfdb.rdsamp(file_path.replace('.dat', ''))
            processed_signals = process_signals(signals, desired_length)
            wavelet_transformed_signals = apply_wavelet_transform(processed_signals)
            all_wavelet_transformed_signals.append(wavelet_transformed_signals)

wavelet_transformed_signals = np.concatenate(all_wavelet_transformed_signals, axis=0)
            
X = np.array(all_wavelet_transformed_signals)
y = np.array(all_labels)
if len(y.shape) > 1:
    y = y[:, 0]

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Design CNN Model
model = models.Sequential()
model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))
