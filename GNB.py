import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score


class GaussianNaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.priors = np.zeros(self.n_classes)
        self.means = np.zeros((self.n_classes, self.n_features))
        self.stds = np.zeros((self.n_classes, self.n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = len(X_c) / len(X)
            self.means[i, :] = X_c.mean(axis=0)
            self.stds[i, :] = X_c.std(axis=0)

    def _pdf(self, class_idx, X):
        mean = self.means[class_idx]
        std = self.stds[class_idx]
        exponent = -1 / 2 * ((X - mean) / std) ** 2
        coefficient = 1 / (std * np.sqrt(2 * np.pi))
        return coefficient * np.exp(exponent)

    def _softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def predict_joint_log_proba(self, X):
        posteriors = []

        for i in range(self.n_classes):
            prior = np.log(self.priors[i])
            class_conditional = np.sum(np.log(self._pdf(i, X)), axis=1)
            posterior = prior + class_conditional
            posteriors.append(posterior)

        posteriors = np.array(posteriors)
        return posteriors.T

    def predict_proba(self, X):
        return self._softmax(self.predict_joint_log_proba(X))

    def predict(self, X):
        y_pred = np.argmax(self.predict_joint_log_proba(X), axis=1)
        return y_pred

    def balanced_accuracy_score(self, X, y):
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)