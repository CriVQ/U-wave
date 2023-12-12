import wfdb
import matplotlib.pyplot as plt

# Set the path to the directory containing the PhysioNet data
data_path = 'data'

# Define the record name (file prefix without extension)
record_name = 'sel100'

# Load the annotation file
annotation = wfdb.rdann(f'{data_path}/{record_name}', extension='q1c')

# Load the corresponding signal file for visualization (optional)
# signal, fields = wfdb.rdsamp(f'{data_path}/{record_name}')

# Plot the annotation on top of the signal (if available)
plt.plot(annotation.sample, [1] * len(annotation.sample), 'ro', label='q1c annotation')
plt.xlabel('Sample')
plt.ylabel('Annotation')
plt.legend()
plt.show()
