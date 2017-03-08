import logging

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class SklearnIntentClassifier(object):
    """Intent classifier using the sklearn framework"""

    def __init__(self, uses_probabilities=True, max_num_threads=1):
        """Construct a new intent classifier using the sklearn framework.

        :param uses_probabilities: defines if the model should be trained
                                           to be able to predict label probabilities
        :param max_num_threads: number of threads used during training time
        :type uses_probabilities: bool"""

        self.le = LabelEncoder()
        self.uses_probabilities = uses_probabilities
        self.tuned_parameters = [{'C': [1, 2, 5, 10, 20, 100], 'kernel': ['linear']}]
        self.score = 'f1'
        self.clf = GridSearchCV(SVC(C=1, probability=uses_probabilities),
                                self.tuned_parameters, n_jobs=max_num_threads,
                                cv=2, scoring='%s_weighted' % self.score)

    def transform_labels_str2num(self, labels):
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation
        :type labels: list of str"""

        y = self.le.fit_transform(labels)
        return y

    def transform_labels_num2str(self, y):
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation
        :type labels: list of str"""

        labels = self.le.inverse_transform(y)
        return labels

    @staticmethod
    def train(intent_examples, featurizer, max_num_threads, test_split_size=0.1):
        """Train the intent classifier on a data set."""
        intent_classifier = SklearnIntentClassifier(max_num_threads=max_num_threads)
        labels = [e["intent"] for e in intent_examples]
        sentences = [e["text"] for e in intent_examples]

        if len(set(labels)) < 2:
            raise Exception("Can not train an intent classifier. Need at least 2 different classes.")
        y = intent_classifier.transform_labels_str2num(labels)
        X = featurizer.features_for_sentences(sentences)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, random_state=0)
        intent_classifier.clf.fit(X_train, y_train)

        # Test the trained model
        if test_split_size != 0.0:
            logging.info("Score of intent model on test data: %s " % intent_classifier.clf.score(X_test, y_test))
        return intent_classifier

    def predict_prob(self, X):
        """Given a bow vector of an input text, predict the intent label. Returns probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""

        if hasattr(self, 'uses_probabilities') and self.uses_probabilities:
            return self.clf.predict_proba(X)
        else:
            y_pred_indices = self.clf.predict(X)
            # convert representation to one-hot. all labels are zero, only the predicted label gets assigned prob=1
            y_pred = np.zeros((np.size(X, 0), len(self.le.classes_)))
            y_pred[np.arange(y_pred.shape[0]), y_pred_indices] = 1
            return y_pred

    def predict(self, X):
        """Given a bow vector of an input text, predict most probable label. Returns only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second, its probability"""

        pred_result = self.predict_prob(X)
        max_indicies = np.argmax(pred_result, axis=1)
        # retrieve the index of the intent with the highest probability
        max_values = pred_result[:, max_indicies].flatten()
        return max_indicies, max_values

    @staticmethod
    def load(path):
        import cloudpickle
        if path:
            with open(path, 'rb') as f:
                return cloudpickle.load(f)
        else:
            return None

    def persist(self, dir_name):
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import os
        import cloudpickle

        classifier_file = os.path.join(dir_name, "intent_classifier.dat")
        with open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "intent_classifier": "intent_classifier.dat"
        }
