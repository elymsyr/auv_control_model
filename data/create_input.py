import h5py
import numpy as np

with h5py.File('data/test_data.h5', 'r') as hf:
    x_current = hf['x_current'][:]  # (num_samples, 12)
    x_ref = hf['x_ref'][:]          # (num_samples, 12*(N+1))

    # Reshape x_ref into 3D array: (samples, horizon+1, state_dim)
    x_path = x_ref.reshape(x_ref.shape[0], 12, 41).transpose(0, 2, 1)

    for i, row in enumerate(x_current):
        pos_array = np.array([row[0], row[1], row[2], 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        x_path[i] -= pos_array

    x_current = x_current[:, 3:]
    X = np.hstack((x_current, x_ref)).astype(np.float32)

# Select the first sample (change index as needed)
sample = X[0]

# Write to input.txt as comma-separated values
np.savetxt('input.txt', sample[None], delimiter=',', fmt='%.6f')
print("Saved first sample to input.txt")