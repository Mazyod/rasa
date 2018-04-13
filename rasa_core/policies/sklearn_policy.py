from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import pickle
import warnings
import typing

from typing import Optional, Any, List, Text, Dict, Callable

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle as sklearn_shuffle

from rasa_core.policies import Policy
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker


class SklearnPolicy(Policy):
    """Use an sklearn classifier to train a policy.

    Supports cross validation and grid search.

    :param sklearn.base.ClassifierMixin model:
      The sklearn model or model pipeline.

    :param cv:
      If *cv* is not None, perform a cross validation on the training
      data. *cv* should then conform to the sklearn standard
      (e.g. *cv=5* for a 5-fold cross-validation).

    :param dict param_grid:
      If *param_grid* is not None and *cv* is given, a grid search on
      the given *param_grid* is performed
      (e.g. *param_grid={'n_estimators': [50, 100]}*).

    :param scoring:
      Scoring strategy, using the sklearn standard.

    :param sklearn.base.TransformerMixin label_encoder:
      Encoder for the labels. Must implement an *inverse_transform*
      method.

    :param bool shuffle:
      Whether to shuffle training data.

    """
    SUPPORTS_ONLINE_TRAINING = True

    def __init__(
            self,
            featurizer=None,  # type: Optional[MaxHistoryTrackerFeaturizer]
            model=LogisticRegression(),  # type: sklearn.base.ClassifierMixin
            param_grid=None,  # type: Optional[Dict[Text, List] or List[Dict]]
            cv=None,  # type: Optional[int]
            scoring='accuracy',  # type: Optional[Text or List or Dict or Callable]
            label_encoder=LabelEncoder(),  # type: sklearn.base.TransformerMixin
            shuffle=True,  # type: bool
    ):
        if featurizer:
            assert isinstance(featurizer, MaxHistoryTrackerFeaturizer), \
                ("Passed featurizer of type {}, should be "
                 "MaxHistoryTrackerFeaturizer."
                 "".format(type(featurizer).__name__))
        super(SklearnPolicy, self).__init__(featurizer)

        self.model = model
        self.cv = cv
        self.param_grid = param_grid
        self.scoring = scoring
        self.label_encoder = label_encoder
        self.shuffle = shuffle

        # attributes that need to be restored after loading
        self._pickle_params = [
            'model', 'cv', 'param_grid', 'scoring', 'label_encoder']

    @property
    def _state(self):
        return {attr: getattr(self, attr) for attr in self._pickle_params}

    def model_architecture(self, **kwargs):
        # filter out kwargs that cannot be passed to model
        params = self._get_valid_params(self.model.__init__, **kwargs)
        return self.model.set_params(**params)

    def _extract_training_data(self, training_data):
        # transform y from one-hot to num_classes
        X, y = training_data.X, training_data.y.argmax(axis=-1)
        if self.shuffle:
            X, y = sklearn_shuffle(X, y)
        return X, y

    def _preprocess_data(self, X, y=None):
        Xt = X.reshape(X.shape[0], -1)
        if y is None:
            return Xt
        else:
            yt = self.label_encoder.transform(y)
            return Xt, yt

    def _search_and_score(self, model, X, y, param_grid):
        search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=self.cv,
            scoring='accuracy',
            verbose=1,
        )
        search.fit(X, y)
        print("Best params:", search.best_params_)
        return search.best_estimator_, search.best_score_

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> Dict[Text: Any]

        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)

        X, y = self._extract_training_data(training_data)
        model = self.model_architecture(**kwargs)
        score = None
        # Note: clone is called throughout to avoid mutating default
        # arguments.
        self.label_encoder = clone(self.label_encoder).fit(y)
        Xt, yt = self._preprocess_data(X, y)

        if self.cv is None:
            model = clone(model).fit(Xt, yt)
        else:
            param_grid = self.param_grid or {}
            model, score = self._search_and_score(
                model, Xt, yt, param_grid)

        self.model = model
        logger.info("Done fitting sklearn policy model")
        if score is not None:
            logger.info("Cross validation score: {:.5f}".format(score))

        return training_data.metadata

    def continue_training(self, training_trackers, domain, **kwargs):
        # type: (list[DialogueStateTracker], Domain, **Any) -> None

        # takes the new example labelled and learns it
        # via taking `epochs` samples of n_batch-1 parts of the training data,
        # inserting our new example and learning them. this means that we can
        # ask the network to fit the example without overemphasising
        # its importance (and therefore throwing off the biases)
        batch_size = kwargs.get('batch_size', 5)
        epochs = kwargs.get('epochs', 50)

        num_samples = batch_size - 1
        num_prev_examples = len(training_trackers) - 1
        for _ in range(epochs):
            sampled_idx = np.random.choice(range(num_prev_examples),
                                           replace=False,
                                           size=min(num_samples,
                                                    num_prev_examples))
            trackers = [training_trackers[i]
                        for i in sampled_idx] + training_trackers[-1:]

            training_data = self.featurize_for_training(trackers,
                                                        domain)
            X, y = self._extract_training_data(training_data)
            Xt, yt = self._preprocess_data(X, y)
            if not hasattr(self.model, 'partial_fit'):
                raise TypeError("Continuing training is only possible with "
                                "sklearn models that support 'partial_fit'.")
            self.model.partial_fit(Xt, yt)

    def _postprocess_prediction(self, y_proba):
        yp = y_proba[0].tolist()

        # Some classes might not be part of the training labels. Since
        # sklearn does not predict labels it has never encountered
        # during training, it is necessary to insert missing classes.
        indices = self.label_encoder.inverse_transform(np.arange(len(yp)))
        y_filled = [0.0 for _ in range(max(indices + 1))]
        for i, pred in zip(indices, yp):
            y_filled[i] = pred

        return y_filled

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        X, _ = self.featurizer.create_X([tracker], domain)
        Xt = self._preprocess_data(X)
        y_proba = self.model.predict_proba(Xt)
        return self._postprocess_prediction(y_proba)

    def persist(self, path):
        # type: (Text) -> None

        super(SklearnPolicy, self).persist(path)

        if not self.model:
            warnings.warn("Persist called without a trained model present. "
                          "Nothing to persist then!")
        else:
            filename = os.path.join(path, 'sklearn_model.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(self._state, f)

    @classmethod
    def load(cls, path):
        # type: (Text) -> Policy
        filename = os.path.join(path, 'sklearn_model.pkl')
        if not os.path.exists(path):
            raise OSError("Failed to load dialogue model. Path {} "
                          "doesn't exist".format(os.path.abspath(filename)))

        featurizer = Featurizer.load(path)
        policy = cls(featurizer=featurizer)

        with open(filename, 'rb') as f:
            state = pickle.load(f)
        vars(policy).update(state)

        logger.info("Loaded sklearn model")
        return policy
