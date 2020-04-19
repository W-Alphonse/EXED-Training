from valeo.infrastructure.LogManager import LogManager

import matplotlib.pyplot as plt
from valeo.infrastructure import Const as C
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score

from valeo.infrastructure.tools.ImgUtil import ImgUtil


class MetricPlotter :
    logger = None

    def __init__(self):
        MetricPlotter.logger = LogManager.logger(__name__);

    def plot_roc(self, y_test, y_pred):
        # y_test = label_binarize(y_test.values, classes=[0, 1])  # y_test 'Series'
        # y_pred = label_binarize(y_pred, classes=[0, 1])         # y_pred  'numpy.ndarray'
        plt.figure()
        lw = 2
        roc = roc_curve(y_test, y_pred)
        plt.plot(roc[0], roc[1], color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc_score(y_test, y_pred))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        ImgUtil.save_fig("ROC_curve")
        plt.show()

    def plot_precision_recall(self, y_test, y_pred):
        average_precision = average_precision_score(y_test, y_pred)
        plt.figure()
        lw = 2
        pr = precision_recall_curve(y_test, y_pred)
        plt.plot(pr[0], pr[1], color='darkorange', lw=lw, label='Precision Recall curve (area = %0.4f)' % average_precision)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve')
        plt.legend(loc="upper right")
        ImgUtil.save_fig("PR_curve")
        plt.show()
        #
        # for i in range(0, len(pr[0]) ) :
        #     print(f"{i}: ({pr[0][i]},{pr[1][i]})")
