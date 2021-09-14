# flake8: noqa
# WARNING: This module will be dropped before Rasa Open Source 3.0 is released.
#          Please don't do any changes in this module and rather adapt
#          SpacyTokenizerGraphComponent from the regular
#          `rasa.nlu.tokenizers.spacy_tokenizer` module. This module is a workaround to
#          defer breaking changes due to the architecture revamp in 3.0.
import typing
from typing import Text, List, Any, Type, Optional

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.components import Component
from rasa.nlu.utils.spacy_utils import SpacyNLP
from rasa.shared.nlu.training_data.message import Message

from rasa.nlu.constants import SPACY_DOCS

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


POS_TAG_KEY = "pos"


class SpacyTokenizer(Tokenizer):
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [SpacyNLP]

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    def get_doc(self, message: Message, attribute: Text) -> Optional["Doc"]:
        return message.get(SPACY_DOCS[attribute])

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        doc = self.get_doc(message, attribute)
        if not doc:
            return []

        tokens = [
            Token(
                t.text, t.idx, lemma=t.lemma_, data={POS_TAG_KEY: self._tag_of_token(t)}
            )
            for t in doc
            if t.text and t.text.strip()
        ]

        return self._apply_token_pattern(tokens)

    @staticmethod
    def _tag_of_token(token: Any) -> Text:
        import spacy

        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_
