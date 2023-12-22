import numpy as np
from sklearn.metrics import confusion_matrix
import pdb

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class BinaryClassificationMetrics(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(self, label_true, label_prob):
        # Convert probabilities to binary predictions using the threshold
        label_pred = (label_prob > self.threshold).astype(np.int32)

        mask_positive = (label_true == 1)
        mask_negative = (label_true == 0)

        # Ensure label_pred and mask_positive are 1-dimensional
        label_pred = label_pred.ravel()
        mask_positive = mask_positive.ravel()

        self.true_positive += np.sum((label_pred[mask_positive.ravel()] == 1))
        self.false_negative += np.sum((label_pred[mask_positive.ravel()] == 0))
        self.true_negative += np.sum((label_pred[mask_negative.ravel()] == 0))
        self.false_positive += np.sum((label_pred[mask_negative.ravel()] == 1))


    def get_results(self):
        precision = self.true_positive / (self.true_positive + self.false_positive + 1e-12)
        recall = self.true_positive / (self.true_positive + self.false_negative + 1e-12)
        accuracy = (self.true_positive + self.true_negative) / (self.true_positive + self.true_negative + self.false_positive + self.false_negative + 1e-12
        )
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-12)

        return {
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1 Score": f1_score,
        }


    def to_str(self, metrics):
        string = "\n"
        for k, v in metrics.items():
            string += "%s: %f\n" % (k, v)
        return string

    def reset(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0


class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
