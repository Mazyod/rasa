import abc
import logging
import re

from typing import Text, List, Dict, Any

from rasa.engine.graph import GraphComponent
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES, MESSAGE_ATTRIBUTES
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_RESPONSE_KEY,
    RESPONSE_IDENTIFIER_DELIMITER,
    ACTION_NAME,
)
from rasa.nlu.tokenizers._tokenizer import Tokenizer, Token

logger = logging.getLogger(__name__)


# This is a workaround around until we have all components migrated to `GraphComponent`.
Tokenizer = Tokenizer
Token = Token


class TokenizerGraphComponent(GraphComponent, abc.ABC):
    """Base class for tokenizers."""

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Construct a new tokenizer."""
        self._config = config
        # flag to check whether to split intents
        self.intent_tokenization_flag = config["intent_tokenization_flag"]
        # split symbol for intents
        self.intent_split_symbol = config["intent_split_symbol"]
        # token pattern to further split tokens
        token_pattern = config["token_pattern"]
        self.token_pattern_regex = None
        if token_pattern:
            self.token_pattern_regex = re.compile(token_pattern)

    @abc.abstractmethod
    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        ...

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Tokenize all training data."""
        for example in training_data.training_examples:
            for attribute in MESSAGE_ATTRIBUTES:
                if (
                    example.get(attribute) is not None
                    and not example.get(attribute) == ""
                ):
                    if attribute in [INTENT, ACTION_NAME, INTENT_RESPONSE_KEY]:
                        tokens = self._split_name(example, attribute)
                    else:
                        tokens = self.tokenize(example, attribute)
                    example.set(TOKENS_NAMES[attribute], tokens)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """Tokenize the incoming messages."""
        for message in messages:
            for attribute in MESSAGE_ATTRIBUTES:
                if isinstance(message.get(attribute), str):
                    if attribute in [
                        INTENT,
                        ACTION_NAME,
                        RESPONSE_IDENTIFIER_DELIMITER,
                    ]:
                        tokens = self._split_name(message, attribute)
                    else:
                        tokens = self.tokenize(message, attribute)

                    message.set(TOKENS_NAMES[attribute], tokens)
        return messages

    def _tokenize_on_split_symbol(self, text: Text) -> List[Text]:
        words = (
            text.split(self.intent_split_symbol)
            if self.intent_tokenization_flag
            else [text]
        )

        return words

    def _split_name(self, message: Message, attribute: Text = INTENT) -> List[Token]:
        text = message.get(attribute)

        # for INTENT_RESPONSE_KEY attribute,
        # first split by RESPONSE_IDENTIFIER_DELIMITER
        if attribute == INTENT_RESPONSE_KEY:
            intent, response_key = text.split(RESPONSE_IDENTIFIER_DELIMITER)
            words = self._tokenize_on_split_symbol(
                intent
            ) + self._tokenize_on_split_symbol(response_key)

        else:
            words = self._tokenize_on_split_symbol(text)

        return self._convert_words_to_tokens(words, text)

    def _apply_token_pattern(self, tokens: List[Token]) -> List[Token]:
        """Apply the token pattern to the given tokens.

        Args:
            tokens: list of tokens to split

        Returns:
            List of tokens.
        """
        if not self.token_pattern_regex:
            return tokens

        final_tokens = []
        for token in tokens:
            new_tokens = self.token_pattern_regex.findall(token.text)
            new_tokens = [t for t in new_tokens if t]

            if not new_tokens:
                final_tokens.append(token)

            running_offset = 0
            for new_token in new_tokens:
                word_offset = token.text.index(new_token, running_offset)
                word_len = len(new_token)
                running_offset = word_offset + word_len
                final_tokens.append(
                    Token(
                        new_token,
                        token.start + word_offset,
                        data=token.data,
                        lemma=token.lemma,
                    )
                )

        return final_tokens

    @staticmethod
    def _convert_words_to_tokens(words: List[Text], text: Text) -> List[Token]:
        running_offset = 0
        tokens = []

        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))

        return tokens
