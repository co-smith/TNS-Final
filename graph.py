import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data saved by run_evaluation.py
cm = np.load('graphs/confusion_matrix_data.npy')

labels = ['Safe', 'Disinfo']

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

plt.ylabel('Actual Ground Truth')
plt.xlabel('Bot Prediction')
plt.title('Final Confusion Matrix')

# Save image
plt.savefig('graphs/confusion_matrix.png')
print("Graph saved to graphs/confusion_matrix.png")