import numpy as np
import matplotlib.pyplot as plt

# Load the true labels and predicted scores
y_true = np.loadtxt("y_true.txt", dtype=float)
y_pred = np.loadtxt("y_pred.txt", dtype=float)

# Separate scores into real (y_true == 1) and fake (y_true == 0)
scores_real = y_pred[y_true == 1]
scores_fake = y_pred[y_true == 0]

# Calculate metrics
mean_real = np.mean(scores_real)
std_dev_real = np.std(scores_real)
mean_fake = np.mean(scores_fake)
std_dev_fake = np.std(scores_fake)

# Find the optimal threshold for classification
thresholds = np.linspace(min(y_pred), max(y_pred), 500)
accuracy = []

for threshold in thresholds:
    predictions = y_pred <= threshold
    acc = np.mean(predictions == y_true)
    accuracy.append(acc)

optimal_threshold = thresholds[np.argmax(accuracy)]
optimal_accuracy = max(accuracy)

# Visualization
plt.figure(figsize=(12, 6))

# Histogram
plt.hist(scores_real, bins=20, alpha=0.7, label="Real Pictures", color="blue", density=True)
plt.hist(scores_fake, bins=20, alpha=0.7, label="Fake Pictures", color="red", density=True)

# Add threshold line
plt.axvline(optimal_threshold, color='green', linestyle='--', label=f"Optimal Threshold: {optimal_threshold:.2f}")

# Labels and legend
plt.title("Score Distribution for Real and Fake Pictures")
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()

# Display metrics
print(f"Metrics for Real Pictures: Mean = {mean_real:.2f}, Std Dev = {std_dev_real:.2f}")
print(f"Metrics for Fake Pictures: Mean = {mean_fake:.2f}, Std Dev = {std_dev_fake:.2f}")
print(f"Optimal Threshold for Classification: {optimal_threshold:.2f}")
print(f"Optimal Accuracy: {optimal_accuracy:.2f}")

plt.show()
