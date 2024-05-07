import matplotlib.pyplot as plt

# Data
iterations = [10, 20, 30, 40, 50, 100, 400]
mse_gpt2 = [0.49660, 0.49210, 0.43770, 0.38619, 0.39356, 0.16834, 0.06944]
mse_gpt2_medium = [0.49660, 0.47313, 0.28428, 0.29078, 0.20753, 0.16655, 0.04699]
mse_gpt2_large = [0.32355, 0.24297, 0.45931, 0.19220, 0.14082, 0.12549, 0.05465]

# Plot
plt.figure(figsize=(10, 6))


plt.axhline(y=0.415, color='purple', linestyle='--', label='Prompt GPT-3')
plt.axhline(y=0.327, color='r', linestyle='--', label='ZeroGPT')

plt.plot(iterations, mse_gpt2, marker='^', label='MedDetect-GPT-small')
plt.plot(iterations, mse_gpt2_medium, marker='^', label='MedDetect-GPT-medium')
plt.plot(iterations, mse_gpt2_large, marker='^', label='MedDetect-GPT-large')

# Add labels and title
plt.xticks([10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400])
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Performance under Different Iterations')
plt.legend()

# Save plot
plt.grid(True)
plt.savefig('mse_plot.png')
