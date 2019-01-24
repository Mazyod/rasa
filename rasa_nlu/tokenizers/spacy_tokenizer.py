import typing
from typing import Any

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Token, Tokenizer
from rasa_nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyTokenizer(Tokenizer, Component):
    name = "tokenizer_spacy"

    provides = ["tokens"]

    requires = ["spacy_doc"]

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any)-> None:

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.get("spacy_doc")))

    def process(self, message: Message, **kwargs: Any)-> None:

        message.set("tokens", self.tokenize(message.get("spacy_doc")))

    def tokenize(self, doc: 'Doc')-> typing.List[Token]:

        return [Token(t.text, t.idx) for t in doc]
