from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from builtins import object
from typing import Any, List, Optional, Text, Dict
from rasa_core.featurizers import \
    MaxHistoryFeaturizer, BinaryFeaturizeMechanism

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.featurizers import Featurizer
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.training.data import DialogueTrainingData

logger = logging.getLogger(__name__)


class Policy(object):
    SUPPORTS_ONLINE_TRAINING = False
    MAX_HISTORY_DEFAULT = 3

    def __init__(self, featurizer=None):
        # type: (Optional[Featurizer]) -> None

        self.featurizer = featurizer

    def prepare(self, featurizer):
        # type: (Featurizer) -> None

        if self.featurizer is None:
            self.featurizer = featurizer
        else:
            logger.warning("Trying to reset featurizer {} "
                           "for policy {} by agent featurizer {}. "
                           "Agent featurizer is ignored."
                           "".format(type(self.featurizer),
                                     type(self),
                                     type(featurizer)))

    @staticmethod
    def _standard_featurizer(max_history=5):
        return MaxHistoryFeaturizer(BinaryFeaturizeMechanism(),
                                    max_history)

    def featurize_for_training(
            self,
            trackers,  # type: List[DialogueStateTracker]
            domain,  # type: Domain
            max_training_samples=None  # type: Optional[int]
    ):
        # type: (...) -> DialogueTrainingData
        """Transform training trackers into a vector representation.
        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model."""

        if self.featurizer is None:
            self.featurizer = self._standard_featurizer()

        training_data, _ = self.featurizer.featurize_trackers(trackers,
                                                              domain)
        if max_training_samples:
            training_data.limit_training_data_to(max_training_samples)

        return training_data

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
        after seeing the tracker.

        Returns the list of probabilities for the next actions"""

        raise NotImplementedError("Policy must have the capacity "
                                  "to predict.")

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> Dict[Text: Any]
        """Trains the policy on given training trackers.

        Returns training metadata."""

        raise NotImplementedError("Policy must have the capacity "
                                  "to train.")

    def continue_training(self, tracker, domain, **kwargs):
        # type: (DialogueStateTracker, Domain, **Any) -> None
        """Continues training an already trained policy.

        This doesn't need to be supported by every policy. If it is supported,
        the policy can be used for online training and the implementation for
        the continued training should be put into this function."""

        pass

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to a storage."""
        self.featurizer.persist(path)

    @classmethod
    def load(cls, path):
        # type: (Text) -> Policy
        """Loads a policy from the storage.

        Needs to load its featurizer"""

        raise NotImplementedError("Policy must have the capacity "
                                  "to load itself.")
