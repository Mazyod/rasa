from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import utilities
from rasa_nlu.training_data import TrainingData, Message


def test_crf_extractor(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    examples = [
        Message("anywhere in the west", {
            "intent": "restaurant_search",
            "entities": [{"start": 16, "end": 20, "value": "west", "entity": "location"}],
            "spacy_doc": spacy_nlp("anywhere in the west")
        }),
        Message("central indian restaurant", {
            "intent": "restaurant_search",
            "entities": [{"start": 0, "end": 7, "value": "central", "entity": "location"}],
            "spacy_doc": spacy_nlp("central indian restaurant")
        })]
    config = {"entity_crf_BILOU_flag": True, "entity_crf_features": ext.crf_features}
    ext.train(TrainingData(training_examples=examples), config)
    sentence = 'anywhere in the west'
    crf_format = ext._from_text_to_crf(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}))
    assert ([word[0] for word in crf_format] == ['anywhere', 'in', 'the', 'west'])
    feats = ext._sentence_to_features(crf_format)
    assert ('BOS' in feats[0])
    assert ('EOS' in feats[-1])
    assert ('0:low:in' in feats[1])
    sentence = 'anywhere in the west'
    ext.extract_entities(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}))


def test_crf_json_from_BILOU(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    ext.BILOU_flag = True
    sentence = u"I need a home cleaning close-by"
    r = ext._from_crf_to_json(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}),
                              ['O', 'O', 'O', 'B-what', 'L-what', 'B-where', 'I-where', 'L-where'])
    assert len(r) == 2, "There should be two entities"
    assert r[0] == {u'start': 9, u'end': 22, u'value': u'home cleaning', u'entity': u'what'}
    assert r[1] == {u'start': 23, u'end': 31, u'value': u'close-by', u'entity': u'where'}


def test_crf_json_from_non_BILOU(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    ext.BILOU_flag = False
    sentence = u"I need a home cleaning close-by"
    r = ext._from_crf_to_json(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}),
                              ['O', 'O', 'O', 'what', 'what', 'where', 'where', 'where'])
    assert len(r) == 5, "There should be five entities"  # non BILOU will split multi-word entities - hence 5
    assert r[0] == {u'start': 9, u'end': 13, u'value': u'home', u'entity': u'what'}
    assert r[1] == {u'start': 14, u'end': 22, u'value': u'cleaning', u'entity': u'what'}
    assert r[2] == {u'start': 23, u'end': 28, u'value': u'close', u'entity': u'where'}
    assert r[3] == {u'start': 28, u'end': 29, u'value': u'-', u'entity': u'where'}
    assert r[4] == {u'start': 29, u'end': 31, u'value': u'by', u'entity': u'where'}


def test_ner_regex_no_entities():
    from rasa_nlu.extractors.regex_entity_extractor import RegExEntityExtractor
    regex_dict = {u'\\bmexican\\b': u'mexican',
                  u'[0-9]+': u'number'}
    txt = "I want indian food"
    ext = RegExEntityExtractor(regex_dict)
    assert ext.extract_entities(txt) == []


def test_ner_regex_multi_entities():
    from rasa_nlu.extractors.regex_entity_extractor import RegExEntityExtractor
    regex_dict = {u'\\bmexican\\b': u'mexican',
                  u'[0-9]+': u'number'}
    txt = "find me 2 mexican restaurants"
    ext = RegExEntityExtractor(regex_dict)
    r = sorted(ext.extract_entities(txt), key=lambda k: k['start'])
    assert r[0] == {u'start': 8, u'end': 9, u'value': '2', u'entity': 'number'}
    assert r[1] == {u'start': 10, u'end': 17, u'value': 'mexican', u'entity': 'mexican'}



def test_ner_regex_1_entity():
    from rasa_nlu.extractors.regex_entity_extractor import RegExEntityExtractor
    regex_dict = {u'\\bmexican\\b': u'mexican',
                  u'[0-9]+': u'number'}
    txt = "my insurance number is 934049430"
    ext = RegExEntityExtractor(regex_dict)
    r = ext.extract_entities(txt)
    assert r[0] == {u'start': 23, u'end': 32, u'value': '934049430', u'entity': 'number'}


def test_duckling_entity_extractor(component_builder):
    _config = utilities.base_test_conf("all_components")
    _config["duckling_dimensions"] = ["time"]
    duckling = component_builder.create_component("ner_duckling", _config)
    message = Message("Today is the 5th of May. Let us meet tomorrow.")
    duckling.process(message)
    entities = message.get("entities")
    assert len(entities) == 3
