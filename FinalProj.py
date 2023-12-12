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
            print(f"Loaded labels for {file}: {labels}")

        elif file_extension == '.hea':
            record = wfdb.rdheader(file_path.replace('.hea', ''))
            print(f"Loaded header for {file}: {record}")

        elif file_extension == '.dat':
            signals, fields = wfdb.rdsamp(file_path.replace('.dat', ''))
            processed_signals = process_signals(signals, desired_length)
            
            wavelet_transformed_signals = apply_wavelet_transform(processed_signals)
            print(f"Wavelet-transformed signals for {file}: {wavelet_transformed_signals}")

            all_wavelet_transformed_signals.append(wavelet_transformed_signals)

wavelet_transformed_signals = np.concatenate(all_wavelet_transformed_signals, axis=0)
            
X = np.array(all_wavelet_transformed_signals)
y = np.array(all_labels)
if len(y.shape) > 1:
    y = y[:, 0]
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Output the number of datapoints for training, validation, and testing
print(f"Number of datapoints for training: {X_train.shape[0]}")
print(f"Number of datapoints for validation: {X_val.shape[0]}")
print(f"Number of datapoints for testing: {X_test.shape[0]}")

# Design CNN Model
model = models.Sequential()
model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with training and validation data
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

predictions = model.predict(X_test)
predicted_labels = np.round(predictions).flatten()

correct_predictions = np.sum(predicted_labels == y_test)
total_samples = len(y_test)

accuracy_per_data_point = correct_predictions / total_samples
print(f'Accuracy per data point on the test set: {accuracy_per_data_point}')


plt.figure(figsize=(10, 6))
plt.bar(range(total_samples), predicted_labels == y_test, color='green', label='Correct Prediction')
plt.xlabel('Data Point')
plt.ylabel('Correct Prediction')
plt.title('Accuracy per Data Point on Test Set')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')

plt.tight_layout()
plt.show()

sample_index =[0,1,2,3,4,5,6,7,8,9,10]

for index in sample_index:

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(X_test[index], label='Original Signal')
    plt.title(f"{index + 1}: Original Signal")
    plt.legend()

  
    plt.subplot(2, 1, 2)
    for i in range(wavelet_transformed_signals.shape[1]):
        plt.plot(wavelet_transformed_signals[:, i], label=f'Lead {i + 1}')
    plt.title(f"{index + 1}: Wavelet Transformed Signals")
    plt.legend()

    plt.tight_layout()
    plt.show()
