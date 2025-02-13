import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define mean and standard deviation
mu = 175  # Mean height
sigma = 10  # Standard deviation

# Generate x values from (mean - 3σ) to (mean + 3σ)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = norm.pdf(x, mu, sigma)

# Set font properties
plt.rcParams["font.family"] = "Courier New"
plt.rcParams["font.size"] = 12

# Plot the Normal Distribution curve
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Normal Distribution", color='blue')

# Add mean and standard deviation markers
plt.axvline(mu, color='red', linestyle='dashed', label="Mean (μ)")
plt.axvline(mu - sigma, color='green', linestyle='dashed', label="1σ Left")
plt.axvline(mu + sigma, color='green', linestyle='dashed', label="1σ Right")

# Labels and title with custom font
plt.title("Normal Distribution (Height of Adult Males)", fontweight='bold')
plt.xlabel("Height (cm)", fontweight='bold')
plt.ylabel("Probability Density", fontweight='bold')
plt.legend()
plt.grid()

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for Standard Normal Distribution
mu = 0      # Mean
sigma = 1   # Standard Deviation

# Generate data points for x-axis (from -4 to 4 to cover most of the distribution)
x = np.linspace(-4, 4, 1000)

# Generate the corresponding y-values (probability density function of the standard normal distribution)
y = norm.pdf(x, mu, sigma)

# Plot the graph
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Standard Normal Distribution', color='b', lw=2)

# Add labels and title
plt.title('Standard Normal Distribution', fontsize=16)
plt.xlabel('Value (X)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)

# Marking the mean (mu) and the standard deviations (sigma)
plt.axvline(mu, color='r', linestyle='--', label=f'Mean (μ = {mu})')
plt.axvline(mu + sigma, color='g', linestyle='--', label=f'1 Standard Deviation (μ + σ)')
plt.axvline(mu - sigma, color='g', linestyle='--', label=f'1 Standard Deviation (μ - σ)')
plt.axvline(mu + 2*sigma, color='orange', linestyle='--', label=f'2 Standard Deviations (μ + 2σ)')
plt.axvline(mu - 2*sigma, color='orange', linestyle='--', label=f'2 Standard Deviations (μ - 2σ)')

# Fill between the curve and x-axis (shading for the standard deviation intervals)
plt.fill_between(x, y, where=(x >= mu - 2*sigma) & (x <= mu + 2*sigma), color='yellow', alpha=0.3, label="95% Coverage (±2σ)")

# Add
# Show grid and plot
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate X values (Z-Scores)
z_scores = np.linspace(-3.5, 3.5, 1000)
pdf_values = stats.norm.pdf(z_scores, 0, 1)  # Standard normal distribution

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(z_scores, pdf_values, label="Standard Normal Distribution", color="blue")

# Shade regions corresponding to standard deviations
plt.fill_between(z_scores, pdf_values, where=(z_scores > -1) & (z_scores < 1), color='blue', alpha=0.3, label="68% (-1σ to +1σ)")
plt.fill_between(z_scores, pdf_values, where=(z_scores > -2) & (z_scores < 2), color='green', alpha=0.2, label="95% (-2σ to +2σ)")
plt.fill_between(z_scores, pdf_values, where=(z_scores > -3) & (z_scores < 3), color='red', alpha=0.1, label="99.7% (-3σ to +3σ)")

# Add labels and title
plt.xlabel("Z-Score")
plt.ylabel("Probability Density")
plt.title("Z-Score Distribution (Standard Normal Curve)")
plt.axvline(0, linestyle="--", color="black", alpha=0.7)  # Mean line at Z=0
plt.legend()
plt.grid()

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set figure size
plt.figure(figsize=(12, 5))

# 1. Log-Normal Distribution
mu, sigma = 2.5, 0.3  # Mean and standard deviation of log(X)
x = np.linspace(0, 50, 1000)  # Range of stock prices
pdf = stats.lognorm.pdf(x, sigma, scale=np.exp(mu))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, label="Log-Normal Distribution", color="blue")
plt.xlabel("Stock Price ($)")
plt.ylabel("Probability Density")
plt.title("Log-Normal Distribution")
plt.legend()
plt.grid()

# 2. Binomial Distribution
n, p = 10, 0.5  # Number of trials, probability of success
x = np.arange(0, n + 1)
pmf = stats.binom.pmf(x, n, p)

plt.subplot(1, 2, 2)
plt.bar(x, pmf, label="Binomial Distribution", color="green", alpha=0.6)
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.title("Binomial Distribution")
plt.legend()
plt.grid()

# Show plots
plt.tight_layout()
plt.show()

# Generate a toy dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate a toy dataset
np.random.seed(42)
class_0 = np.random.normal(loc=[155, 55], scale=[5, 5], size=(10, 2))  # Short people
class_1 = np.random.normal(loc=[175, 75], scale=[5, 5], size=(10, 2))  # Tall people

# Labels
labels_0 = np.zeros(len(class_0))
labels_1 = np.ones(len(class_1))

# Combine data
data = np.vstack((class_0, class_1))
labels = np.hstack((labels_0, labels_1))

# Query point
query_point = np.array([[165, 60]])

# Fit KNN model
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(data, labels)
predicted_class = knn.predict(query_point)[0]

# Find the k nearest neighbors
distances, indices = knn.kneighbors(query_point)

# Plot dataset
plt.figure(figsize=(8, 6), facecolor='lightgrey')
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0 (Short)', edgecolors='black', alpha=0.7)
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1 (Tall)', edgecolors='black', alpha=0.7)
plt.scatter(query_point[0, 0], query_point[0, 1], color='green', s=150, marker='X', label='Query Point')

# Highlight the nearest neighbors
plt.scatter(data[indices][0][:, 0], data[indices][0][:, 1], 
            s=200, facecolors='none', edgecolors='gold', linewidths=2, label=f'Nearest {k} Neighbors')

# Labels and title
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("K-Nearest Neighbors (KNN) Visualization")
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# Given dataset: (Age, Salary)
points = [(150, None), (60, None), (47, None), (300, None), (70, 110), (85, 140)]

# Sort by Age (x-axis) for KD-Tree construction
sorted_points = sorted(points, key=lambda x: x[0])

# KD-Tree structure based on median split
tree_structure = {
    "root": (85, 140),
    "left": {
        "node": (60, None),
        "left": (47, None),
        "right": (70, 110),
    },
    "right": {
        "node": (150, None),
        "right": (300, None)
    }
}

# Plot KD-Tree structure
fig, ax = plt.subplots(figsize=(6, 6))

# Draw tree connections
ax.plot([85, 60], [140, 120], 'k-', lw=1.5)  # Root to Left
ax.plot([85, 150], [140, 130], 'k-', lw=1.5)  # Root to Right
ax.plot([60, 47], [120, 110], 'k-', lw=1.5)  # Left child to its left
ax.plot([60, 70], [120, 110], 'k-', lw=1.5)  # Left child to its right
ax.plot([150, 300], [130, 125], 'k-', lw=1.5)  # Right child to its right

# Scatter plot of nodes
for point in points:
    ax.scatter(point[0], 120 if point[1] is None else point[1], s=100, color="blue")

# Annotate nodes
ax.text(85, 140, "(85,140)", fontsize=12, ha='right', color='red', fontweight='bold')
ax.text(60, 120, "(60,?)", fontsize=12, ha='right', color='black')
ax.text(47, 110, "(47,?)", fontsize=12, ha='right', color='black')
ax.text(70, 110, "(70,110)", fontsize=12, ha='right', color='black')
ax.text(150, 130, "(150,?)", fontsize=12, ha='right', color='black')
ax.text(300, 125, "(300,?)", fontsize=12, ha='right', color='black')

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("KD-Tree Visualization")
ax.set_xlabel("Age (X-axis)")
ax.set_ylabel("Salary (Y-axis)")

plt.grid(False)
plt.show()

