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

    desert_values = sum(1 for i in y_pred if 0.3 < i < 0.7)  # predictions in the interval (0.3, 0.7)

    return y_true, y_pred, desert_values

def plot_calibration(y_true, y_pred):
    for strategy in ('uniform', 'quantile'):
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy=strategy)
        print(prob_true)

        brier_score = brier_score_loss(y_true, y_pred)
        print(f'Brier score: {brier_score}')

        plt.style.use('fivethirtyeight')

        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax1.plot(prob_pred, prob_true, 's-', label='Model\'s calibration')

        if strategy == 'uniform':
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            counts, bins, _ = ax2.hist(y_pred, range=(0, 1), bins=10, histtype='bar', rwidth=0.7)
            for n, b in zip(counts, bins):
                if n > 0:
                    ax2.text(b+0.038, n, str(int(n)))
            ax2.set_ylabel('Count')
            ax2.set_xlabel('Confidence')

        ax1.axvspan(0.3, 0.7, color='gray', alpha=0.5, label='No Man\'s Land')  # grey rectangle
        ax1.legend(loc="lower right")
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability curve')

        #plt.yscale('log')
        #plt.tight_layout()

        plt.show()
