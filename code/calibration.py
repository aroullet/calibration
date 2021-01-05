import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# TODO: plot reliability diagrams for blast cells detection
# TODO: plot for each class

classes_dictionary_org = {'BAS': 0, 'EBO': 1, 'EOS': 2, 'KSC': 3, 'LYA': 4, 'LYT': 5, 'MMZ': 6, 'MOB': 7,
                          'MON': 8, 'MYB': 9, 'MYO': 10, 'NGB': 11, 'NGS': 12, 'PMB': 13, 'PMO': 14}
classes_dictionary = {value: key for key, value in classes_dictionary_org.items()}


class CalibrationCurves:

    def __init__(self, preds, files):
        self.preds = preds
        self.files = files

        print(self.files)

    def extract_probs(self, cell=None):
        y_true = []
        y_pred = []

        def all_probs():
            for i in range(len(self.files)):
                for j in range(len(self.preds[i])):
                    y_pred.append(self.preds[i][j])

                    if self.files[i][:3] == classes_dictionary[j]:
                        y_true.append(1)
                    else:
                        y_true.append(0)

            assert len(y_true) == len(y_pred)
            return y_true, y_pred

        def class_probs():
            index = classes_dictionary_org[cell]

            for i in range(len(self.files)):
                for j in range(len(self.preds[i])):
                    if j == index:
                        y_pred.append(self.preds[i][j])

                        if self.files[i][:3] == cell:
                            y_true.append(1)
                        else:
                            y_true.append(0)

            assert len(y_true) == len(y_pred)
            return y_true, y_pred

        if cell is None:
            return all_probs()
        else:
            return class_probs()

    def plot(self, y_true, y_pred):
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        print('Accuracy per bin: ', prob_true)

        brier_score = brier_score_loss(y_true, y_pred)

        plt.style.use('seaborn')

        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        ax1.plot(prob_pred, prob_true, 's-', label='Model\'s calibration')
        ax1.set_ylim([-0.05, 1.05])
        ax1.text(0.4, 0.2, f'Brier Score: {round(brier_score, 3)}')

        ax1.axvspan(0.2, 0.8, color='gray', alpha=0.4, label='No Man\'s Land')  # grey rectangle
        ax1.legend(loc='lower right')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability curve')

        ax2 = plt.subplot2grid((3, 1), (2, 0))
        counts, bins, _ = ax2.hist(y_pred, range=(0, 1), bins=10, histtype='bar', rwidth=0.7)
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Confidence')
        ax2.set_yscale('log')
        print('Bins and counts: ', list(zip(bins, counts)))

        plt.savefig('C:/Users/roull/Documents/Calibration/results/Figure1.png', dpi=300)

        plt.tight_layout()
        plt.show()

        print(f'Brier score: {brier_score}')
