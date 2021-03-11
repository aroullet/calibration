import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import uncertainty_metrics.numpy as um

classes_dictionary_org = {'BAS': 0, 'EBO': 1, 'EOS': 2, 'KSC': 3, 'LYA': 4, 'LYT': 5, 'MMZ': 6, 'MOB': 7,
                          'MON': 8, 'MYB': 9, 'MYO': 10, 'NGB': 11, 'NGS': 12, 'PMB': 13, 'PMO': 14}

classes_dictionary = {value: key for key, value in classes_dictionary_org.items()}


class CalibrationCurves:

    def __init__(self, preds, files, n):
        self.preds = preds
        self.files = files
        self.n = n

    def get_probs(self, cell=None):
        y_true = np.empty(shape=(self.n,), dtype=int)
        y_pred = np.empty(shape=(self.n,))

        def _all_probs():
            y_true = np.empty(shape=(self.n, 15,), dtype=int)
            y_pred = np.empty(shape=(self.n, 15))

            for i in range(len(self.files)):
                y_pred[i] = self.preds[i]
                for j in range(15):
                    if self.files[i][:3] == classes_dictionary[j]:
                        y_true[i][j] = 1
                    else:
                        y_true[i][j] = 0

            # ensure all values < 1 without creating a copy of the array (because EBO class is sum of 2 classes)
            np.minimum(y_pred, 1, out=y_pred)
            return y_true, y_pred

        def _class_probs():
            index = classes_dictionary_org[cell]

            for i in range(len(self.files)):
                y_pred[i] = self.preds[i][index]
                for j in range(len(self.preds[i])):
                    if self.files[i][:3] == cell:
                        y_true[i] = 1
                    else:
                        y_true[i] = 0

            return y_true, y_pred

        def _blast_probs():
            for i in range(len(self.files)):
                y_pred[i] = self.preds[i][7] + self.preds[i][10]
                if self.files[i][:3] in ('MYO', 'MOB'):  # blast cell codes
                    y_true[i] = 1
                else:
                    y_true[i] = 0

            return y_true, y_pred

        def _atypical_probs():
            for i in range(len(self.files)):
                total_prob = sum(self.preds[i][j] for j in (1, 4, 6, 7, 9, 10, 14))
                y_pred[i] = total_prob
                if self.files[i][:3] in ('MYO', 'MOB', 'MYB', 'MMZ', 'PMO', 'EBO', 'LYA'):  # atypical cell codes
                    y_true[i] = 1
                else:
                    y_true[i] = 0

            np.minimum(y_pred, 1, out=y_pred)
            return y_true, y_pred

        if cell is None:
            return _all_probs()
        elif cell in classes_dictionary_org:
            return _class_probs()
        elif cell == 'blast':
            return _blast_probs()
        elif cell == 'atypical':
            return _atypical_probs()
        else:
            raise ValueError(f"'cell' should be one of the cell codes, 'atypical', 'blast' or None. Got {cell}. ")

    def plot(self, y_true, y_pred):
        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        print('Accuracy per bin: ', prob_true)
        print('Confidence per bin: ', prob_pred)

        brier_score = brier_score_loss(y_true, y_pred)
        print(f'Brier score: {brier_score}')

        ece = um.gce(y_true, y_pred, binning_scheme='even', max_prob=False,
                     class_conditional=False, norm='l1', num_bins=10)
        print('ECE: ', ece)

        plt.style.use('seaborn')
        plt.rc('font', size=18)
        plt.rc('axes', titlesize=22)
        plt.rc('axes', labelsize=18)

        import pylab as plot
        params = {'legend.fontsize': 14}
        plot.rcParams.update(params)

        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        ax1.plot(prob_pred, prob_true, 's-', label='Model\'s calibration')
        ax1.set_ylim([-0.05, 1.05])
        ax1.text(0, 0.8, f'Brier Score: {round(brier_score, 3)}')
        ax1.text(0, 0.7, f'ECE : {round(ece, 3)}')

        ax1.legend(loc='lower right')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability curve (global)')

        ax2 = plt.subplot2grid((3, 1), (2, 0))
        counts, bins, _ = ax2.hist(y_pred, range=(0, 1), bins=10, histtype='bar', rwidth=0.7)

        ax2.set_ylabel('Count')
        ax2.set_xlabel('Confidence')
        ax2.set_yscale('log')

        print('Bins and counts: ', list(zip(bins, counts)))

        plt.tight_layout()
        plt.show()


# Useful metric for an overview of model calibration
def multi_class_brier_score(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 15)
    y_pred = np.array(y_pred).reshape(-1, 15)
    return np.mean(np.sum((y_true-y_pred) ** 2, axis=1))
