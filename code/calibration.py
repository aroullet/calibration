import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import numpy as np

def extract_probs(preds, files, dct):
    y_true = []
    y_pred = []

    for i in range(len(files)):
        index_max = np.argmax(preds[i])
        for j in range(len(preds[i])):
            y_pred.append(preds[i][j])

            if j == index_max and files[i][:3] == dct[index_max]:
                y_true.append(1)
            else:
                y_true.append(0)
    
    assert len(y_true) == len(y_pred)
    return y_true, y_pred

def plot_calibration(y_true, y_pred):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    brier_score = brier_score_loss(y_true, y_pred)
    print(f'Brier score: {brier_score}')

    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2, fig=fig)
    ax2 = plt.subplot2grid((3, 1), (2, 0), fig=fig)

    ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax1.plot(prob_pred, prob_true, 's-', label='Model\'s calibration')
    ax2.hist(y_pred, range=(0, 1), bins=10, histtype="step", lw=2)

    ax1.set_ylabel('Accuracy')
    ax1.legend(loc="lower right")
    ax1.set_title('Reliability curve')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Confidence')

    plt.tight_layout()

    plt.show()
