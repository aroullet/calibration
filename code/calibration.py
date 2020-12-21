import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def extract_probs(preds, files, dct):
    y_true = []
    y_pred = []

    for i in range(len(files)):
        for j in range(len(preds[i])):
            y_pred.append(preds[i][j])

            if files[i][:3] == dct[j]:
                y_true.append(1)
            else:
                y_true.append(0)

    desert_values = sum(1 for i in y_pred if 0.3 < i < 0.7)  # predictions in the interval (0.3, 0.7)

    return y_true, y_pred, desert_values

def plot_calibration(y_true, y_pred):
    for strategy in ('uniform', 'quantile'):
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy=strategy)
        print('Accuracy per bin: ', prob_true)

        brier_score = brier_score_loss(y_true, y_pred)

        plt.style.use('seaborn')

        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        ax1.plot(prob_pred, prob_true, 's-', label='Model\'s calibration')
        ax1.set_ylim([-0.05, 1.05])
        ax1.text(0.4, 0.2, f'Brier Score: {round(brier_score, 3)}')

        if strategy == 'uniform':
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            counts, bins, _ = ax2.hist(y_pred, range=(0, 1), bins=10, histtype='bar', rwidth=0.7)

            print('Bins and counts: ', list(zip(bins, counts)))
            ax2.set_ylabel('Count')
            ax2.set_xlabel('Confidence')
            ax2.set_yscale('log')

        if strategy == 'quantile':
            ax1.set_xlabel('Confidence')

        ax1.axvspan(0.3, 0.7, color='gray', alpha=0.4, label='No Man\'s Land')  # grey rectangle
        ax1.legend(loc="lower right")
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability curve')

        plt.tight_layout()

        plt.show()

    print(f'Brier score: {brier_score}')
