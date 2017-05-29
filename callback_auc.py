from collections import defaultdict
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class CallbackAUC(Callback):
    def __init__(self, name, peptides, mhc_seqs, labels, weights, predictor):
        Callback.__init__(self)
        self.name = name
        self.peptides = peptides
        self.mhc_seqs = mhc_seqs
        self.labels = labels
        self.weights = weights
        self.predictor = predictor
        self.aucs = defaultdict(list)

    def on_epoch_end(self, epoch, logs):
        Y_pred_dict = self.predictor.predict_scores({
            "peptide": self.peptides,
            "mhc": self.mhc_seqs})
        for output_name, y in Y_pred_dict.items():
            auc = roc_auc_score(
                y_true=self.labels,
                y_score=y,
                sample_weight=self.weights)
            print("--> %s epoch %d: AUC=%0.4f" % (
                self.name,
                epoch,
                auc))
            self.aucs[output_name].append(auc)
