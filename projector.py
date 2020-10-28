import tensorflow as tf
import keras.backend as K
from keras.utils import to_categorical

class ProjectorClassifier():
    def __init__(self):
        self.labels = None
        self._isfitted = False
        self._entropy = tf.keras.losses.KLDivergence()
        #self._entropy = tf.keras.losses.MeanSquaredError()

    def fit(self, X, Y):
        classes, _ = tf.unique(Y)
        self.labels = tf.map_fn(lambda c: tf.reduce_mean(tf.boolean_mask(X,tf.equal(Y, tf.repeat(c, tf.constant(Y.shape[0], dtype=tf.int64)))), axis=0), classes, dtype=tf.float64)
        self._isfitted = True

    def predict(self, X):
        if not self._isfitted:
            raise ValueError("Call Projector.fit() first")
        predictions = tf.argmin(tf.map_fn(lambda label: tf.map_fn(lambda x: self._entropy(x, label), X), self.labels), axis=0)
        return predictions

    def evaluate(self, X, Y):
        if not self._isfitted:
            raise ValueError("Call Projector.fit() first")
        Y_pred = self.predict(X)
        if tf.reduce_max(Y) > tf.constant(1, dtype=tf.int64):
            Y = to_categorical(Y)
            Y_pred = to_categorical(Y_pred)
        acc = accuracy_score(Y, Y_pred)
        auc = roc_auc_score(Y, Y_pred, average="weighted", multi_class="ovr")
        print(f"Accuracy: {acc}, AUC:{auc}")
        return [acc, auc]
