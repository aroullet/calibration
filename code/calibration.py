import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np

def extract_probs(preds, files, dct):
    y_true = []
    y_prob = []

    for i in range(len(files)):
        index_max = np.argmax(preds[i])
        for j in range(len(preds[i])):
            y_prob.append(preds[i][j])

            if j == index_max and files[i][:3] == dct[index_max]:
                y_true.append(1)
            else:
                y_true.append(0)
    
    assert len(y_true) == len(y_prob)
    return y_true, y_prob

def plot_calibration():
    a, b = calibration_curve(y_true, y_prob)