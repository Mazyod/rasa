import re
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message
from rasa.nlu.constants import INTENT_ATTRIBUTE, TOKENS_NAMES, MESSAGE_ATTRIBUTES


class WhitespaceTokenizer(Tokenizer):

    provides = [TOKENS_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES]

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Text will be tokenized with case sensitive as default
        "case_sensitive": True,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)
        # flag to check whether to split intents
        self.intent_tokenization_flag = self.component_config.get(
            "intent_tokenization_flag"
        )
        # split symbol for intents
        self.intent_split_symbol = self.component_config["intent_split_symbol"]
        self.case_sensitive = self.component_config["case_sensitive"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        if not self.case_sensitive:
            text = text.lower()

        if attribute != INTENT_ATTRIBUTE:
            # remove 'not a word character' if
            words = re.sub(
                # there is a space or an end of a string after it
                r"[^\w#@&]+(?=\s|$)|"
                # there is a space or beginning of a string before it
                # not followed by a number
                r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
                # not in between numbers and not . or @ or & or - or #
                # e.g. 10'000.00 or blabla@gmail.com
                # and not url characters
                r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
                " ",
                text,
            ).split()
            # Fix for convert tokenizer, remove any single '&'
            words = [w for w in words if w != "&"]
            # if we removed everything like smiles `:)`, use the whole text as 1 token
            if not words:
                words = [text]
        else:
            words = (
                text.split(self.intent_split_symbol)
                if self.intent_tokenization_flag
                else [text]
            )

        running_offset = 0
        tokens = []

        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))

        return tokens
