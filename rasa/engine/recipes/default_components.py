from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.nlu.classifiers.logistic_regression_classifier import (
    LogisticRegressionClassifier,
)
from rasa.nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.extractors.duckling_entity_extractor import DucklingEntityExtractor
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

DEFAULT_COMPONENTS = [
    # Message Classifiers
    DIETClassifier,
    FallbackClassifier,
    KeywordIntentClassifier,
    SklearnIntentClassifier,
    LogisticRegressionClassifier,
    # Message Entity Extractors
    CRFEntityExtractor,
    DucklingEntityExtractor,
    EntitySynonymMapper,
    RegexEntityExtractor,
    # Message Feauturizers
    LexicalSyntacticFeaturizer,
    ConveRTFeaturizer,
    CountVectorsFeaturizer,
    RegexFeaturizer,
    # Tokenizers
    WhitespaceTokenizer,
]
