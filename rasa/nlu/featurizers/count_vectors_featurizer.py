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

logger = logging.getLogger(__name__)

# TODO: Check if we want to support shared vocab
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

        self.use_shared_vocab = self.component_config['use_shared_vocab']

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
        vectorizer: Optional["CountVectorizer"] = None,
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

        if self.vectorizer is not None:
            if isinstance(self.vectorizer, list):
                self.vectorizer[0].tokenizer = lambda x: self._tokenizer(self, x)
                self.vectorizer[1].tokenizer = lambda x: self._tokenizer(self, x)
            else:
                self.vectorizer.tokenizer = lambda x: self._tokenizer(self, x)


    @staticmethod
    def _tokenizer(self, text):
        """Override tokenizer in CountVectorizer."""

        text = re.sub(r"\b[0-9]+\b", "__NUMBER__", text)

        token_pattern = re.compile(self.token_pattern)
        tokens = token_pattern.findall(text)

        if self.OOV_token:
            if hasattr(self.vectorizer, "vocabulary_"):
                # CountVectorizer is trained, process for prediction
                if self.OOV_token in self.vectorizer.vocabulary_:
                    tokens = [
                        t if t in self.vectorizer.vocabulary_.keys() else self.OOV_token
                        for t in tokens
                    ]
            elif self.OOV_words:
                # CountVectorizer is not trained, process for train
                tokens = [self.OOV_token if t in self.OOV_words else t for t in tokens]

        return tokens

    @staticmethod
    def _get_message_text(message):
        if message.get("spacy_doc"):  # if lemmatize is possible
            return " ".join([t.lemma_ for t in message.get("spacy_doc")])
        elif message.get("tokens"):  # if directly tokens is provided
            return " ".join([t.text for t in message.get("tokens")])
        else:
            return message.text

    @staticmethod
    def _get_message_intent(message):
        return ' '.join([t.text for t in message.get("intent_tokens")])

    @staticmethod
    def _get_message_response(message):
        if message.get("response_tokens") is None:
            return ' '
        return ' '.join([t.text for t in message.get("response_tokens")])

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
            self.vectorizer = CountVectorizer(token_pattern=self.token_pattern,
                                        strip_accents=self.strip_accents,
                                        lowercase=self.lowercase,
                                        stop_words=self.stop_words,
                                        ngram_range=(self.min_ngram,
                                                     self.max_ngram),
                                        max_df=self.max_df,
                                        min_df=self.min_df,
                                        max_features=self.max_features,
                                        tokenizer=lambda x: self._tokenizer(self,x),
                                        analyzer=self.analyzer)
        else:
            self.vectorizer = [CountVectorizer(token_pattern=self.token_pattern,
                                         strip_accents=self.strip_accents,
                                         lowercase=self.lowercase,
                                         stop_words=self.stop_words,
                                         ngram_range=(self.min_ngram,
                                                      self.max_ngram),
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         max_features=self.max_features,
                                         tokenizer=lambda x: self._tokenizer(self,x),
                                         analyzer=self.analyzer),
                         CountVectorizer(token_pattern=self.token_pattern,
                                         strip_accents=self.strip_accents,
                                         lowercase=self.lowercase,
                                         stop_words=self.stop_words,
                                         ngram_range=(self.min_ngram,
                                                      self.max_ngram),
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         max_features=self.max_features,
                                         tokenizer=lambda x: self._tokenizer(self,x),
                                         analyzer=self.analyzer)]

        lem_exs = [
            self._get_message_text(example) for example in training_data.intent_examples
        ]

        self._check_OOV_present(lem_exs)

        lem_ints = [self._get_message_intent(example)
                    for example in training_data.intent_examples]

        lem_resps = [self._get_message_response(example)
                     for example in training_data.intent_examples]

        empty_resps = False
        unique_resps = set(lem_resps)
        if len(unique_resps) == 1 and next(iter(unique_resps)) == ' ':
            empty_resps = True

        self._check_OOV_present(lem_ints)
        self._check_OOV_present(lem_resps)

        # noinspection PyPep8Naming
        try:
            if self.use_shared_vocab:
                # noinspection PyPep8Naming
                self.vectorizer.fit(lem_exs + lem_ints + lem_resps)
                X = self.vectorizer.transform(lem_exs).toarray()
                Y_ints = self.vectorizer.transform(lem_ints).toarray()
                Y_resps = self.vectorizer.transform(lem_resps)

            else:
                X = self.vectorizer[0].fit_transform(lem_exs).toarray()
                Y_ints = self.vectorizer[1].fit_transform(lem_ints).toarray()
                if not empty_resps:
                    Y_resps = self.vectorizer[1].fit_transform(lem_resps).toarray()
                else:
                    Y_resps = None

        except ValueError:
            self.vectorizer = None
            return

        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set(
                "text_features",
                self._combine_with_existing_text_features(example, X[i]),
            )
            example.set("intent_features", Y_ints[i])
            if Y_resps is not None:
                example.set("response_features", Y_resps[i])

    def process(self, message: Message, **kwargs: Any) -> None:

        # TODO: add featurization of test data

        if self.vectorizer is None:
            logger.error(
                "There is no trained CountVectorizer: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
        else:
            message_text = self._get_message_text(message)

            if self.use_shared_vocab:
                bag = self.vectorizer.transform([message_text]).toarray().squeeze()
            else:
                bag = self.vectorizer[0].transform([message_text]).toarray().squeeze()
            message.set(
                "text_features", self._combine_with_existing_text_features(message, bag)
            )

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again.
        """

        file_name = file_name + ".pkl"
        if self.vectorizer:
            featurizer_file = os.path.join(model_dir, file_name)
            if not self.use_shared_vocab:
                utils.json_pickle(featurizer_file,[self.vectorizer[0].vocabulary_, self.vectorizer[1].vocabulary_])
            else:
                utils.json_pickle(featurizer_file, self.vectorizer.vocabulary_)
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

            if isinstance(vocabulary,list):
                vectorizer = [CountVectorizer(token_pattern=meta["token_pattern"],
                                                strip_accents=meta["strip_accents"],
                                                lowercase=meta["lowercase"],
                                                stop_words=meta["stop_words"],
                                                ngram_range=(meta["min_ngram"], meta["max_ngram"]),
                                                max_df=meta["max_df"],
                                                min_df=meta["min_df"],
                                                max_features=meta["max_features"],
                                                analyzer=meta["analyzer"],
                                                vocabulary=vocabulary[0],
                                              ),
                              CountVectorizer(token_pattern=meta["token_pattern"],
                                                strip_accents=meta["strip_accents"],
                                                lowercase=meta["lowercase"],
                                                stop_words=meta["stop_words"],
                                                ngram_range=(meta["min_ngram"], meta["max_ngram"]),
                                                max_df=meta["max_df"],
                                                min_df=meta["min_df"],
                                                max_features=meta["max_features"],
                                                analyzer=meta["analyzer"],
                                                vocabulary=vocabulary[1],
                                              )
                              ]
            else:
                vectorizer = CountVectorizer(token_pattern=meta["token_pattern"],
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

            return cls(meta, vectorizer)
        else:
            return cls(meta)
