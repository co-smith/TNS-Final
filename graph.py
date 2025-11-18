import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array([[80, 8], [21, 33]]) 
labels = ['Safe', 'Disinfo']

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Ground Truth')
plt.xlabel('Bot Prediction')
plt.title('Final Confusion Matrix (Optimized Model)')
plt.savefig('graphs/confusion_matrix.png')
print("Chart saved as confusion_matrix.png")