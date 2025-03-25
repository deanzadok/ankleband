import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a Pandas DataFrame
data_dir = '/Data/imugr/datasets/subjects/prepared_data'
data = pd.read_hdf(path_or_buf=os.path.join(data_dir, 'ID07_standing_free.h5'), key='df')

# chosen seconds
start_sec = 10
end_sec = 20

# [dx, dy, dz, dth, dph, dps]
data_columns = data.columns[-6:].tolist()
data['time_x'] = data['time_x'].values - data['time_x'].iloc[0]
data = data[data['time_x'] >= start_sec]
data = data[data['time_x'] <= end_sec]
timestamp = data['time_x'].values
x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro = data[data_columns].values.T

# Create a figure with six subplots
fig, axs = plt.subplots(6, 1, figsize=(10, 15))

# Plot each DOF on a separate subplot
axs[0].plot(timestamp, x_acc)
axs[0].set_ylabel('X Acceleration')
axs[0].set_xlim([timestamp[0], timestamp[-1]])

axs[1].plot(timestamp, y_acc)
axs[1].set_ylabel('Y Acceleration')
axs[1].set_xlim([timestamp[0], timestamp[-1]])

axs[2].plot(timestamp, z_acc)
axs[2].set_ylabel('Z Acceleration')
axs[2].set_xlim([timestamp[0], timestamp[-1]])

axs[3].plot(timestamp, x_gyro)
axs[3].set_ylabel('X Gyroscope')
axs[3].set_xlim([timestamp[0], timestamp[-1]])

axs[4].plot(timestamp, y_gyro)
axs[4].set_ylabel('Y Gyroscope')
axs[4].set_xlim([timestamp[0], timestamp[-1]])

axs[5].plot(timestamp, z_gyro)
axs[5].set_ylabel('Z Gyroscope')
axs[5].set_xlabel('Time')
axs[5].set_xlim([timestamp[0], timestamp[-1]])

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('imu_sample.png')