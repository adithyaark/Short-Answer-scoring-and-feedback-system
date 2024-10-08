import pandas as pd
import matplotlib.pyplot as plt

predicted_scores_both = [
    2.56,
    4.465,
    4.32,
    4.75,
    4.21,
    2.135,
    2.735,
    4.305,
    3.72,
    4.365,
    5.0,
    5.0,
    2.965,
    4.21,
    2.54,
    4.035,
    5.0,
    2.445,
    3.81,
    2.63,
    5.0,
    5.0,
    3.31,
    3.48,
    5.0,
    2.62,
    3.26,
    2.725,
    2.89,
    3.47,
    3.74,
    4.32,
    2.885,
    3.96,
    2.14,
    2.64,
    3.645,
    2.66,
    4.71,
    2.8,
    3.675,
    2.3,
    3.415,
    2.25,
    4.415,
    3.31,
    2.085,
    2.85,
    2.03,
    3.01,
    4.75,
    3.345,
    1.805,
    2.685,
    3.505,
    2.57,
    2.98,
    1.915,
    3.99,
    4.73,
    3.275,
    2.135,
    3.34,
    1.51,
    3.715,
    3.12,
    2.91,
    3.545,
    4.085,
    3.57,
    3.015,
    3.395,
    4.25,
    4.405,
    4.145,
    3.505,
    3.795,
    4.38,
    3.555,
    3.575,
    4.325,
    3.59,
    3.555,
    3.875,
]

predicted_scores_lexical = [
    2.695,
    4.405,
    4.39,
    4.9,
    3.135,
    2.065,
    2.38,
    4.47,
    3.585,
    4.79,
    5.0,
    5.0,
    3.57,
    4.435,
    2.835,
    3.865,
    5.0,
    2.215,
    4.47,
    3.095,
    5.0,
    5.0,
    4.38,
    3.135,
    5.0,
    2.45,
    3.685,
    3.275,
    2.65,
    3.59,
    4.015,
    4.41,
    3.57,
    4.475,
    2.15,
    3.131746295371295,
    4.045,
    3.131746295371295,
    4.545,
    2.22,
    4.14,
    3.131746295371295,
    3.131746295371295,
    2.21,
    4.465,
    3.495,
    2.125,
    3.131746295371295,
    2.665,
    3.131746295371295,
    4.545,
    3.131746295371295,
    2.045,
    2.8,
    4.12,
    3.57,
    3.025,
    1.91,
    1.41,
    4.6,
    3.615,
    2.265,
    3.005,
    1.38,
    3.131746295371295,
    4.61,
    3.055,
    3.7,
    3.845,
    3.565,
    3.131746295371295,
    3.131746295371295,
    4.075,
    3.7,
    4.12,
    3.62,
    3.131746295371295,
    4.25,
    3.57,
    3.57,
    4.315,
    3.57,
    3.57,
    3.155,
]

predicted_scores_semantic = [
    3.94,
    3.995,
    4.175,
    4.64,
    2.43,
    1.98,
    2.555,
    4.13,
    2.96,
    3.865,
    5.0,
    5.0,
    2.905,
    4.445,
    2.425,
    4.36,
    5.0,
    2.99,
    3.165,
    2.985,
    5.0,
    5.0,
    3.995,
    3.105,
    5.0,
    2.995,
    3.21,
    3.315,
    2.83,
    3.31,
    3.74,
    3.885,
    2.865,
    3.89,
    1.98,
    2.45,
    2.965,
    2.56,
    4.64,
    2.205,
    2.99,
    2.46,
    3.4,
    1.77,
    4.47,
    2.69,
    2.09,
    3.18,
    2.165,
    2.99,
    4.28,
    3.855,
    2.355,
    2.28,
    3.265,
    2.96,
    2.82,
    1.895,
    4.605,
    4.605,
    3.085,
    2.13,
    3.18,
    1.47,
    4.21,
    2.56,
    3.015,
    3.555,
    4.24,
    3.465,
    4.175,
    3.34,
    4.175,
    4.605,
    4.035,
    3.52,
    3.98,
    4.0,
    3.52,
    3.435,
    3.715,
    3.17,
    3.52,
    3.87,
]

test = pd.read_csv("dataset-small.csv")
actual_scores = list(test["score"])

x_values = list(range(1, len(actual_scores) + 1))

# Plot actual scores
plt.plot(x_values, actual_scores, color="blue", label="Actual Scores")

# Plot predicted scores
plt.plot(x_values, predicted_scores_semantic, color="red", label="Predicted Scores")

# Add labels and title
plt.xlabel("Index")
plt.ylabel("Scores")
plt.title("Actual vs Predicted Scores")

plt.suptitle("Model Using Semantic Features")

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()
