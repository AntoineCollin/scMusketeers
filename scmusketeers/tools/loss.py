import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K


class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, similarity, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.similarity = similarity  # similarity is a 1D tensor indicating if X and X_perm belong to the same class (0 => diff classes, 1 => same classes)
        self.margin = margin

    def call(self, X_perm, X_pred):
        X_perm = tf.cast(X_perm, X_pred.dtype)
        self.similarity = tf.cast(self.similarity, X_pred.dtype)

        D = tf.square(X_perm - X_pred)
        max_D = tf.square(tf.maximum(0.0, self.margin - D))
        l = self.similarity * D + (1 - self.similarity) * max_D
        return l


class WeightedMSE(tf.keras.losses.Loss):
    def __init__(self, weights, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.similarity = weights  # similarity is a 1D tensor indicating if X and X_perm belong to the same class (0 => diff classes, 1 => same classes)
        self.margin = margin

    def call(self, X_perm, X_pred):
        X_perm = tf.cast(X_perm, X_pred.dtype)
        self.similarity = tf.cast(self.similarity, X_pred.dtype)

        D = tf.square(X_perm - X_pred)
        max_D = tf.square(tf.maximum(0.0, self.margin - D))
        l = self.similarity * D + (1 - self.similarity) * max_D
        return l


class BatchRemoval_loss(tf.keras.losses.Loss):
    def __init__(self, similarity, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.similarity = similarity  # similarity is a 1D tensor indicating if X and X_perm belong to the same class (0 => diff classes, 1 => same classes)
        self.margin = margin

    def call(self, X_perm, X_pred):
        X_perm = tf.cast(X_perm, X_pred.dtype)
        self.similarity = tf.cast(self.similarity, X_pred.dtype)

        D = tf.square(X_perm - X_pred)
        max_D = tf.square(tf.maximum(0.0, self.margin - D))
        l = self.similarity * D + (1 - self.similarity) * max_D
        return l
