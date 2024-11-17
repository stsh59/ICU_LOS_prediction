import matplotlib.pyplot as plt
import numpy as np

# Serial timings (in seconds)
serial_data_preprocessing = (
    0.1824350357055664  # Load Data
    + 0.0271451473236084  # Process Length of Stay
    + 0.0014340877532958984  # Process Death
    + 0.0369563102722168  # Process Ethnicity
    + 0.0014448165893554688  # Process Religion
    + 0.011281013488769531  # Process Admissions
    + 1.1848509311676025  # Process ICD Codes
    + 0.01993703842163086  # Process Patients
    + 0.03128409385681152  # Process Age
    + 0.7841930389404297  # Process ICU
    + 0.2792646884918213  # Preprocess Data
)
serial_model_training = 30.705135822296143  # Model Training

# Parallel timings (in seconds)
parallel_data_preprocessing = (
    1.0314528942108154  # Data Processing
    + 1.0128149032592773  # Merging and Processing Data
    + 0.0808100700378418  # Preprocessing Data
)
parallel_model_training = (
    8.305623054504395  # Model Training
    + 0.0010409355163574219  # Evaluation
)

# Calculate percentage times relative to serial execution
serial_percentage = [100, 100]  # Serial execution is the baseline (100%)
parallel_percentage = [
    (parallel_data_preprocessing / serial_data_preprocessing) * 100,
    (parallel_model_training / serial_model_training) * 100,
]

# Data for plotting
categories = ["Data Preprocessing and Processing", "Model Training and Prediction"]
x = np.arange(len(categories))  # Label locations
bar_width = 0.35  # Width of bars

# Plot grouped bar chart
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width / 2, serial_percentage, bar_width, label='Serial', color='blue')
plt.bar(x + bar_width / 2, parallel_percentage, bar_width, label='Parallel', color='orange')

# Add labels and title
plt.ylabel("Execution Time (% of Serial Time)")
plt.title("Percentage Comparison: Serial vs Parallel (Updated Timings)")
plt.xticks(x, categories)
plt.axhline(100, color='black', linewidth=0.8, linestyle='--', label="Baseline (100%)")
plt.legend()

# Save the plot as an image
plt.tight_layout()
plt.savefig("serial_vs_parallel_comparison.png")
print("Chart saved as 'serial_vs_parallel_comparison.png'.")

# Display the plot
plt.show()