import numpy as np

from nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    MESSAGE_TOKENS_NAMES,
)
from rasa.nlu.training_data import Message
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.config import RasaNLUModelConfig


def test_convert_featurizer(mitie_feature_extractor, default_config):
    from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import (
        ConveRTFeaturizer,
    )

    component_config = {"name": "ConveRTFeaturizer", "return_sequence": False}
    featurizer = ConveRTFeaturizer.create(component_config, RasaNLUModelConfig())

    sentence = "Hey how are you today ?"
    message = Message(sentence)
    tokens = WhitespaceTokenizer().tokenize(sentence)
    message.set(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE], tokens)

    featurizer.process(message)

    vecs = message.get(MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE])[0]

    expected = np.array([1.0251294, -0.04053932, -0.7018805, -0.82054937, -0.75054353])

    assert np.allclose(vecs[:5], expected, atol=1e-5)
