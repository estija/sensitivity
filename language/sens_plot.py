import numpy as np
import matplotlib.pyplot as plt

# Function to compute average and standard deviation for the last 100 rows
def compute_stats(data):
    if data.shape[0] >= 100:
        last_100_rows = data[-100:]
        average = np.mean(last_100_rows, axis=0)
        std_deviation = np.sqrt(np.var(last_100_rows, axis=0))  # Standard deviation
        return average, std_deviation
    else:
        return None, None

# Load the data
lstm_sens_pos = np.load("logs/qqp/lstm/variance-4/sensitivity.npy")
roberta_sens_pos = np.load("logs/qqp/roberta-scratch/variance-4/sensitivity.npy")
roberta_relu_sens_pos = np.load("logs/qqp/roberta-scratch_relu/variance-4/sensitivity.npy")

# Compute stats for each dataset
lstm_avg, lstm_std = compute_stats(lstm_sens_pos)
roberta_avg, roberta_std = compute_stats(roberta_sens_pos)
roberta_relu_avg, roberta_relu_std = compute_stats(roberta_relu_sens_pos)

# Plotting
plt.figure(figsize=(12, 6))

# Check if the data was sufficient and plot
if lstm_avg is not None and lstm_std is not None:
    plt.plot(lstm_avg, label='LSTM Average')
    plt.fill_between(range(len(lstm_avg)), lstm_avg - lstm_std, lstm_avg + lstm_std, 
                     color='purple', alpha=0.3)

if roberta_avg is not None and roberta_std is not None:
    plt.plot(roberta_avg, label='Roberta Average')
    plt.fill_between(range(len(roberta_avg)), roberta_avg - roberta_std, roberta_avg + roberta_std, 
                     color='green', alpha=0.3)

if roberta_relu_avg is not None and roberta_relu_std is not None:
    plt.plot(roberta_relu_avg, label='Roberta ReLU Average')
    plt.fill_between(range(len(roberta_relu_avg)), roberta_relu_avg - roberta_relu_std, roberta_relu_avg + roberta_relu_std, 
                     color='blue', alpha=0.3)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Average and Variance of Last 100 Rows for LSTM, Roberta, and Roberta ReLU')
plt.legend()
plt.savefig('plot.png') 
