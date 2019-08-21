import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

from sklearn.feature_extraction.text import CountVectorizer
from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers import Featurizer
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.utils.train_utils import create_vectorizer

logger = logging.getLogger(__name__)

from rasa.nlu.constants import (
    MESSAGE_ATTRIBUTES,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_RESPONSE_ATTRIBUTE,
)


class CountVectorsFeaturizer(Featurizer):
    """Bag of words featurizer

    Creates bag-of-words representation of intent features
    using sklearn's `CountVectorizer`.
    All tokens which consist only of digits (e.g. 123 and 99
    but not ab12d) will be represented by a single feature.

    Set `analyzer` to 'char_wb'
    to use the idea of Subword Semantic Hashing
    from https://arxiv.org/abs/1810.07150.
    """

    provides = ["text_features"]

    requires = []

    defaults = {
        # whether to use a shared vocab
        "use_shared_vocab": False,
        # the parameters are taken from
        # sklearn's CountVectorizer
        # whether to use word or character n-grams
        # 'char_wb' creates character n-grams inside word boundaries
        # n-grams at the edges of words are padded with space.
        "analyzer": "word",  # use 'char' or 'char_wb' for character
        # regular expression for tokens
        # only used if analyzer == 'word'
        "token_pattern": r"(?u)\b\w\w+\b",
        # remove accents during the preprocessing step
        "strip_accents": None,  # {'ascii', 'unicode', None}
        # list of stop words
        "stop_words": None,  # string {'english'}, list, or None (default)
        # min document frequency of a word to add to vocabulary
        # float - the parameter represents a proportion of documents
        # integer - absolute counts
        "min_df": 1,  # float in range [0.0, 1.0] or int
        # max document frequency of a word to add to vocabulary
        # float - the parameter represents a proportion of documents
        # integer - absolute counts
        "max_df": 1.0,  # float in range [0.0, 1.0] or int
        # set range of ngrams to be extracted
        "min_ngram": 1,  # int
        "max_ngram": 1,  # int
        # limit vocabulary size
        "max_features": None,  # int or None
        # if convert all characters to lowercase
        "lowercase": True,  # bool
        # handling Out-Of-Vacabulary (OOV) words
        # will be converted to lowercase if lowercase is True
        "OOV_token": None,  # string or None
        "OOV_words": [],  # string or list of strings
    }

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    def _load_count_vect_params(self):

        self.use_shared_vocab = self.component_config["use_shared_vocab"]

        # set analyzer
        self.analyzer = self.component_config["analyzer"]

        # regular expression for tokens
        self.token_pattern = self.component_config["token_pattern"]

        # remove accents during the preprocessing step
        self.strip_accents = self.component_config["strip_accents"]

        # list of stop words
        self.stop_words = self.component_config["stop_words"]

        # min number of word occurancies in the document to add to vocabulary
        self.min_df = self.component_config["min_df"]

        # max number (fraction if float) of word occurancies
        # in the document to add to vocabulary
        self.max_df = self.component_config["max_df"]

        # set ngram range
        self.min_ngram = self.component_config["min_ngram"]
        self.max_ngram = self.component_config["max_ngram"]

        # limit vocabulary size
        self.max_features = self.component_config["max_features"]

        # if convert all characters to lowercase
        self.lowercase = self.component_config["lowercase"]

        # Flag to check if test data has been featurized or not.
        self.is_test_data_featurized = False

    # noinspection PyPep8Naming
    def _load_OOV_params(self):
        self.OOV_token = self.component_config["OOV_token"]

        self.OOV_words = self.component_config["OOV_words"]
        if self.OOV_words and not self.OOV_token:
            logger.error(
                "The list OOV_words={} was given, but "
                "OOV_token was not. OOV words are ignored."
                "".format(self.OOV_words)
            )
            self.OOV_words = []

        if self.lowercase and self.OOV_token:
            # convert to lowercase
            self.OOV_token = self.OOV_token.lower()
            if self.OOV_words:
                self.OOV_words = [w.lower() for w in self.OOV_words]

    def _check_analyzer(self):
        if self.analyzer != "word":
            if self.OOV_token is not None:
                logger.warning(
                    "Analyzer is set to character, "
                    "provided OOV word token will be ignored."
                )
            if self.stop_words is not None:
                logger.warning(
                    "Analyzer is set to character, "
                    "provided stop words will be ignored."
                )
            if self.max_ngram == 1:
                logger.warning(
                    "Analyzer is set to character, "
                    "but max n-gram is set to 1. "
                    "It means that the vocabulary will "
                    "contain single letters only."
                )

    def __init__(
        self,
        component_config: Dict[Text, Any] = None,
        vectorizer: Optional[Dict[Text, "CountVectorizer"]] = None,
    ) -> None:
        """Construct a new count vectorizer using the sklearn framework."""

        super(CountVectorsFeaturizer, self).__init__(component_config)

        # parameters for sklearn's CountVectorizer
        self._load_count_vect_params()

        # handling Out-Of-Vocabulary (OOV) words
        self._load_OOV_params()

        # warn that some of config parameters might be ignored
        self._check_analyzer()

        # declare class instance for CountVectorizer
        self.vectorizer = vectorizer

    def _get_message_text_by_attribute(self, message, attribute="text"):

        attribute = "" if attribute == "text" else attribute + "_"
        if message.get("{0}spacy_doc".format(attribute)):  # if lemmatize is possible
            tokens = [t.lemma_ for t in message.get("{0}spacy_doc".format(attribute))]
        elif message.get(
            "{0}tokens".format(attribute)
        ):  # if directly tokens is provided
            tokens = [t.text for t in message.get("{0}tokens".format(attribute))]
        else:
            return ""

        text = re.sub(r"\b[0-9]+\b", "__NUMBER__", " ".join(tokens))
        if self.lowercase:
            text = text.lower()

        if self.OOV_token and self.analyzer == "word":
            text_tokens = text.split()
            if hasattr(self.vectorizer[attribute], "vocabulary_"):
                # CountVectorizer is trained, process for prediction
                if self.OOV_token in self.vectorizer[attribute].vocabulary_:
                    text_tokens = [
                        t
                        if t in self.vectorizer[attribute].vocabulary_.keys()
                        else self.OOV_token
                        for t in text_tokens
                    ]
            elif self.OOV_words:
                # CountVectorizer is not trained, process for train
                text_tokens = [
                    self.OOV_token if t in self.OOV_words else t for t in text_tokens
                ]
            text = " ".join(text_tokens)

        return text

    @staticmethod
    def _get_text_sequence(text):
        return text.split()

    # noinspection PyPep8Naming
    def _check_OOV_present(self, examples):
        if self.OOV_token and not self.OOV_words:
            for t in examples:
                if self.OOV_token in t or (
                    self.lowercase and self.OOV_token in t.lower()
                ):
                    return

            logger.warning(
                "OOV_token='{}' was given, but it is not present "
                "in the training data. All unseen words "
                "will be ignored during prediction."
                "".format(self.OOV_token)
            )

    def train(
        self, training_data: TrainingData, cfg: RasaNLUModelConfig = None, **kwargs: Any
    ) -> None:
        """Train the featurizer.

        Take parameters from config and
        construct a new count vectorizer using the sklearn framework.
        """

        spacy_nlp = kwargs.get("spacy_nlp")
        if spacy_nlp is not None:
            # create spacy lemma_ for OOV_words
            self.OOV_words = [t.lemma_ for w in self.OOV_words for t in spacy_nlp(w)]

        if self.use_shared_vocab:
            shared_vectorizer = create_vectorizer(
                self.token_pattern,
                self.strip_accents,
                self.lowercase,
                self.stop_words,
                (self.min_ngram, self.max_ngram),
                self.max_df,
                self.min_df,
                self.max_features,
                self.analyzer,
            )
            self.vectorizer = {
                attribute: shared_vectorizer for attribute in MESSAGE_ATTRIBUTES
            }
        else:
            self.vectorizer = {
                attribute: create_vectorizer(
                    self.token_pattern,
                    self.strip_accents,
                    self.lowercase,
                    self.stop_words,
                    (self.min_ngram, self.max_ngram),
                    self.max_df,
                    self.min_df,
                    self.max_features,
                    self.analyzer,
                )
                for attribute in MESSAGE_ATTRIBUTES
            }

        cleaned_attribute_texts = {}

        for attribute in MESSAGE_ATTRIBUTES:
            lem = [
                self._get_message_text_by_attribute(example, attribute)
                for example in training_data.intent_examples
            ]
            self._check_OOV_present(lem)
            cleaned_attribute_texts[attribute] = lem

        combined_cleaned_texts = []
        if self.use_shared_vocab:
            for attribute in MESSAGE_ATTRIBUTES:
                combined_cleaned_texts += cleaned_attribute_texts[attribute]

        featurized_attributes = {}
        # noinspection PyPep8Naming
        try:
            for attribute in MESSAGE_ATTRIBUTES:
                if self.use_shared_vocab:
                    self.vectorizer[attribute].fit(combined_cleaned_texts)
                else:
                    self.vectorizer[attribute].fit(cleaned_attribute_texts[attribute])

                featurized_attributes[attribute] = (
                    self.vectorizer[attribute]
                    .transform(cleaned_attribute_texts[attribute])
                    .toarray()
                )

                print (attribute, self.vectorizer[attribute].vocabulary_)

        except ValueError:
            self.vectorizer = None
            return

        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example

            for attribute in MESSAGE_ATTRIBUTES:
                if len(cleaned_attribute_texts[attribute][i]):
                    example.set(
                        "{0}_features".format(attribute),
                        self._combine_with_existing_features(
                            example, featurized_attributes[attribute][i], attribute
                        ),
                    )

    def process(self, message: Message, **kwargs: Any) -> None:

        # TODO: add featurization of test data

        if self.vectorizer is None:
            logger.error(
                "There is no trained CountVectorizer: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
        else:
            message_text = self._get_message_text_by_attribute(
                message, attribute=MESSAGE_TEXT_ATTRIBUTE
            )

            bag = (
                self.vectorizer[MESSAGE_TEXT_ATTRIBUTE]
                .transform([message_text])
                .toarray()
                .squeeze()
            )
            message.set(
                "{0}_features".format(MESSAGE_TEXT_ATTRIBUTE),
                self._combine_with_existing_features(
                    message, bag, attribute=MESSAGE_TEXT_ATTRIBUTE
                ),
            )

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again.
        """

        file_name = file_name + ".pkl"
        if self.vectorizer:
            featurizer_file = os.path.join(model_dir, file_name)
            if not self.use_shared_vocab:
                utils.json_pickle(
                    featurizer_file,
                    [
                        self.vectorizer[attribute].vocabulary_
                        for attribute in MESSAGE_ATTRIBUTES
                    ],
                )
            else:
                utils.json_pickle(
                    featurizer_file, self.vectorizer[MESSAGE_TEXT_ATTRIBUTE].vocabulary_
                )
        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["CountVectorsFeaturizer"] = None,
        **kwargs: Any
    ) -> "CountVectorsFeaturizer":

        file_name = meta.get("file")
        featurizer_file = os.path.join(model_dir, file_name)

        if os.path.exists(featurizer_file):
            vocabulary = utils.json_unpickle(featurizer_file)

            if isinstance(vocabulary, list):
                vectorizer = {
                    attribute: create_vectorizer(
                        token_pattern=meta["token_pattern"],
                        strip_accents=meta["strip_accents"],
                        lowercase=meta["lowercase"],
                        stop_words=meta["stop_words"],
                        ngram_range=(meta["min_ngram"], meta["max_ngram"]),
                        max_df=meta["max_df"],
                        min_df=meta["min_df"],
                        max_features=meta["max_features"],
                        analyzer=meta["analyzer"],
                        vocabulary=vocabulary[index],
                    )
                    for index, attribute in enumerate(MESSAGE_ATTRIBUTES)
                }

            else:
                shared_vectorizer = create_vectorizer(
                    token_pattern=meta["token_pattern"],
                    strip_accents=meta["strip_accents"],
                    lowercase=meta["lowercase"],
                    stop_words=meta["stop_words"],
                    ngram_range=(meta["min_ngram"], meta["max_ngram"]),
                    max_df=meta["max_df"],
                    min_df=meta["min_df"],
                    max_features=meta["max_features"],
                    analyzer=meta["analyzer"],
                    vocabulary=vocabulary,
                )

                vectorizer = {
                    attribute: shared_vectorizer for attribute in MESSAGE_ATTRIBUTES
                }

            return cls(meta, vectorizer)
        else:
            return cls(meta)
