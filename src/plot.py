
import matplotlib.pyplot as plt
import numpy as np


def confusion_matrix(cm, labels):
  plt.imshow(cm, cmap='Blues')
  plt.title('Confusion matrix')
  # plt.colorbar()

  tick_marks = np.arange(len(labels))
  plt.xticks(tick_marks, labels, rotation=45)
  plt.yticks(tick_marks, labels)
  plt.xlim(-0.5, len(labels) - 0.5)
  plt.ylim(len(labels) - 0.5, -0.5)
  plt.grid(linewidth=0.2)

  cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  