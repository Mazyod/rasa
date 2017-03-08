from rasa_nlu.utils.mitie import MITIE_SKLEARN_BACKEND_NAME

from extractors.mitie_entity_extractor import MITIEEntityExtractor
from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
from rasa_nlu.trainers.trainer import Trainer


class MITIESklearnTrainer(Trainer):
    SUPPORTED_LANGUAGES = {"en"}

    name = MITIE_SKLEARN_BACKEND_NAME

    def __init__(self, fe_file, language_name, max_num_threads=1):
        super(self.__class__, self).__init__(language_name, max_num_threads)
        self.fe_file = fe_file
        self.featurizer = MITIEFeaturizer.load(self.fe_file)

    def train_entity_extractor(self, entity_examples):
        self.entity_extractor = MITIEEntityExtractor.train(entity_examples, self.fe_file, self.max_num_threads)

    def train_intent_classifier(self, intent_examples, test_split_size=0.1):
        self.intent_classifier = SklearnIntentClassifier.train(intent_examples,
                                                               self.featurizer,
                                                               self.max_num_threads,
                                                               test_split_size)
