import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def get_data(preds, files):
    y_true = []
    y_prob = []

    for i in range(len(files)):
        for j in range(15):
            y_prob.append(preds[i][j])

            if preds[i][j] == np.argmax(preds[i]) and files[i][:3] == classes_dictionary[np.argmax(preds[i])]:
                y_true.append(0)
            else:
                y_true.append(1)
    
    assert len(y_true) == len(y_prob)
    return y_true, y_prob
    