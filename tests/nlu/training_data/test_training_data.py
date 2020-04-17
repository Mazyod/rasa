from typing import Optional, Text

import pytest
import tempfile
from jsonschema import ValidationError

from rasa.nlu.constants import TEXT, RESPONSE_KEY_ATTRIBUTE
from rasa.nlu import training_data
from rasa.nlu.convert import convert_training_data
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.training_data import TrainingData
from rasa.nlu.training_data.formats import MarkdownReader, MarkdownWriter
from rasa.nlu.training_data.formats.rasa import validate_rasa_nlu_data, RasaReader
from rasa.nlu.training_data.loading import guess_format, UNK, load_data
from rasa.nlu.training_data.util import get_file_format
import rasa.utils.io as io_utils


def test_example_training_data_is_valid():
    demo_json = "data/examples/rasa/demo-rasa.json"
    data = io_utils.read_json_file(demo_json)
    validate_rasa_nlu_data(data)


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"wrong_top_level": []},
        ["this is not a toplevel dict"],
        {
            "rasa_nlu_data": {
                "common_examples": [{"intent": "some example without text"}]
            }
        },
        {
            "rasa_nlu_data": {
                "common_examples": [
                    {
                        "text": "mytext",
                        "entities": [{"start": "INVALID", "end": 0, "entity": "x"}],
                    }
                ]
            }
        },
    ],
)
def test_validation_is_throwing_exceptions(invalid_data):
    with pytest.raises(ValidationError):
        validate_rasa_nlu_data(invalid_data)


def test_luis_data():
    td = training_data.load_data("data/examples/luis/demo-restaurants_v5.json")

    assert not td.is_empty()
    assert len(td.entity_examples) == 8
    assert len(td.intent_examples) == 28
    assert len(td.training_examples) == 28
    assert td.entity_synonyms == {}
    assert td.intents == {"affirm", "goodbye", "greet", "inform"}
    assert td.entities == {"location", "cuisine"}


def test_wit_data():
    td = training_data.load_data("data/examples/wit/demo-flights.json")
    assert not td.is_empty()
    assert len(td.entity_examples) == 4
    assert len(td.intent_examples) == 1
    assert len(td.training_examples) == 4
    assert td.entity_synonyms == {}
    assert td.intents == {"flight_booking"}
    assert td.entities == {"location", "datetime"}


def test_dialogflow_data():
    td = training_data.load_data("data/examples/dialogflow/")
    assert not td.is_empty()
    assert len(td.entity_examples) == 5
    assert len(td.intent_examples) == 24
    assert len(td.training_examples) == 24
    assert len(td.lookup_tables) == 2
    assert td.intents == {"affirm", "goodbye", "hi", "inform"}
    assert td.entities == {"cuisine", "location"}
    non_trivial_synonyms = {k: v for k, v in td.entity_synonyms.items() if k != v}
    assert non_trivial_synonyms == {
        "mexico": "mexican",
        "china": "chinese",
        "india": "indian",
    }
    # The order changes based on different computers hence the grouping
    assert {td.lookup_tables[0]["name"], td.lookup_tables[1]["name"]} == {
        "location",
        "cuisine",
    }
    assert {
        len(td.lookup_tables[0]["elements"]),
        len(td.lookup_tables[1]["elements"]),
    } == {4, 6}


def test_lookup_table_json():
    lookup_fname = "data/test/lookup_tables/plates.txt"
    td_lookup = training_data.load_data("data/test/lookup_tables/lookup_table.json")
    assert not td_lookup.is_empty()
    assert td_lookup.lookup_tables[0]["name"] == "plates"
    assert td_lookup.lookup_tables[0]["elements"] == lookup_fname
    assert td_lookup.lookup_tables[1]["name"] == "drinks"
    assert td_lookup.lookup_tables[1]["elements"] == [
        "mojito",
        "lemonade",
        "sweet berry wine",
        "tea",
        "club mate",
    ]


def test_lookup_table_md():
    lookup_fname = "data/test/lookup_tables/plates.txt"
    td_lookup = training_data.load_data("data/test/lookup_tables/lookup_table.md")
    assert not td_lookup.is_empty()
    assert td_lookup.lookup_tables[0]["name"] == "plates"
    assert td_lookup.lookup_tables[0]["elements"] == lookup_fname
    assert td_lookup.lookup_tables[1]["name"] == "drinks"
    assert td_lookup.lookup_tables[1]["elements"] == [
        "mojito",
        "lemonade",
        "sweet berry wine",
        "tea",
        "club mate",
    ]


@pytest.mark.parametrize(
    "files",
    [
        [
            "data/examples/rasa/demo-rasa.json",
            "data/examples/rasa/demo-rasa-responses.md",
        ],
        [
            "data/examples/rasa/demo-rasa.md",
            "data/examples/rasa/demo-rasa-responses.md",
        ],
    ],
)
def test_demo_data(files):
    from rasa.importers.utils import training_data_from_paths

    td = training_data_from_paths(files, language="en")
    assert td.intents == {"affirm", "greet", "restaurant_search", "goodbye", "chitchat"}
    assert td.entities == {"location", "cuisine"}
    assert td.responses == {"I am Mr. Bot", "It's sunny where I live"}
    assert len(td.training_examples) == 46
    assert len(td.intent_examples) == 46
    assert len(td.response_examples) == 4
    assert len(td.entity_examples) == 11
    assert len(td.nlg_stories) == 2

    assert td.entity_synonyms == {
        "Chines": "chinese",
        "Chinese": "chinese",
        "chines": "chinese",
        "vegg": "vegetarian",
        "veggie": "vegetarian",
    }

    assert td.regex_features == [
        {"name": "greet", "pattern": r"hey[^\s]*"},
        {"name": "zipcode", "pattern": r"[0-9]{5}"},
    ]


@pytest.mark.parametrize(
    "files",
    [
        [
            "data/examples/rasa/demo-rasa.json",
            "data/examples/rasa/demo-rasa-responses.md",
        ],
        [
            "data/examples/rasa/demo-rasa.md",
            "data/examples/rasa/demo-rasa-responses.md",
        ],
    ],
)
def test_demo_data_filter_out_retrieval_intents(files):
    from rasa.importers.utils import training_data_from_paths

    td = training_data_from_paths(files, language="en")
    assert len(td.training_examples) == 46

    td1 = td.filter_training_examples(lambda ex: ex.get(RESPONSE_KEY_ATTRIBUTE) is None)
    assert len(td1.training_examples) == 42

    td2 = td.filter_training_examples(
        lambda ex: ex.get(RESPONSE_KEY_ATTRIBUTE) is not None
    )
    assert len(td2.training_examples) == 4

    # make sure filtering operation doesn't mutate the source training data
    assert len(td.training_examples) == 46


@pytest.mark.parametrize(
    "filepaths",
    [["data/examples/rasa/demo-rasa.md", "data/examples/rasa/demo-rasa-responses.md"]],
)
def test_train_test_split(filepaths):
    from rasa.importers.utils import training_data_from_paths

    td = training_data_from_paths(filepaths, language="en")
    assert td.intents == {"affirm", "greet", "restaurant_search", "goodbye", "chitchat"}
    assert td.entities == {"location", "cuisine"}
    assert len(td.training_examples) == 46
    assert len(td.intent_examples) == 46

    td_train, td_test = td.train_test_split(train_frac=0.8)

    assert len(td_train.training_examples) == 35
    assert len(td_test.training_examples) == 11


@pytest.mark.parametrize(
    "filepaths",
    [["data/examples/rasa/demo-rasa.md", "data/examples/rasa/demo-rasa-responses.md"]],
)
def test_train_test_split_with_random_seed(filepaths):
    from rasa.importers.utils import training_data_from_paths

    td = training_data_from_paths(filepaths, language="en")

    td_train_1, td_test_1 = td.train_test_split(train_frac=0.8, random_seed=1)
    td_train_2, td_test_2 = td.train_test_split(train_frac=0.8, random_seed=1)
    train_1_intent_examples = [e.text for e in td_train_1.intent_examples]
    train_2_intent_examples = [e.text for e in td_train_2.intent_examples]

    test_1_intent_examples = [e.text for e in td_test_1.intent_examples]
    test_2_intent_examples = [e.text for e in td_test_2.intent_examples]

    assert train_1_intent_examples == train_2_intent_examples
    assert test_1_intent_examples == test_2_intent_examples


@pytest.mark.parametrize(
    "files",
    [
        ("data/examples/rasa/demo-rasa.json", "data/test/multiple_files_json"),
        ("data/examples/rasa/demo-rasa.md", "data/test/multiple_files_markdown"),
        ("data/examples/rasa/demo-rasa.md", "data/test/duplicate_intents_markdown"),
    ],
)
def test_data_merging(files):
    td_reference = training_data.load_data(files[0])
    td = training_data.load_data(files[1])
    assert len(td.entity_examples) == len(td_reference.entity_examples)
    assert len(td.intent_examples) == len(td_reference.intent_examples)
    assert len(td.training_examples) == len(td_reference.training_examples)
    assert td.intents == td_reference.intents
    assert td.entities == td_reference.entities
    assert td.entity_synonyms == td_reference.entity_synonyms
    assert td.regex_features == td_reference.regex_features


def test_markdown_single_sections():
    td_regex_only = training_data.load_data(
        "data/test/markdown_single_sections/regex_only.md"
    )
    assert td_regex_only.regex_features == [{"name": "greet", "pattern": r"hey[^\s]*"}]

    td_syn_only = training_data.load_data(
        "data/test/markdown_single_sections/synonyms_only.md"
    )
    assert td_syn_only.entity_synonyms == {"Chines": "chinese", "Chinese": "chinese"}


def test_repeated_entities():
    data = """
{
  "rasa_nlu_data": {
    "common_examples" : [
      {
        "text": "book a table today from 3 to 6 for 3 people",
        "intent": "unk",
        "entities": [
          {
            "entity": "description",
            "start": 35,
            "end": 36,
            "value": "3"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode(io_utils.DEFAULT_ENCODING))
        f.flush()
        td = training_data.load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example.get("entities")
        assert len(entities) == 1
        tokens = WhitespaceTokenizer().tokenize(example, attribute=TEXT)
        start, end = MitieEntityExtractor.find_entity(entities[0], example.text, tokens)
        assert start == 9
        assert end == 10


def test_multiword_entities():
    data = """
{
  "rasa_nlu_data": {
    "common_examples" : [
      {
        "text": "show me flights to New York City",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 32,
            "value": "New York City"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode(io_utils.DEFAULT_ENCODING))
        f.flush()
        td = training_data.load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example.get("entities")
        assert len(entities) == 1
        tokens = WhitespaceTokenizer().tokenize(example, attribute=TEXT)
        start, end = MitieEntityExtractor.find_entity(entities[0], example.text, tokens)
        assert start == 4
        assert end == 7


def test_nonascii_entities():
    data = """
{
  "luis_schema_version": "5.0",
  "utterances" : [
    {
      "text": "I am looking for a ßäæ ?€ö) item",
      "intent": "unk",
      "entities": [
        {
          "entity": "description",
          "startPos": 19,
          "endPos": 26
        }
      ]
    }
  ]
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode(io_utils.DEFAULT_ENCODING))
        f.flush()
        td = training_data.load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example.get("entities")
        assert len(entities) == 1
        entity = entities[0]
        assert entity["value"] == "ßäæ ?€ö)"
        assert entity["start"] == 19
        assert entity["end"] == 27
        assert entity["entity"] == "description"


def test_entities_synonyms():
    data = """
{
  "rasa_nlu_data": {
    "entity_synonyms": [
      {
        "value": "nyc",
        "synonyms": ["New York City", "nyc", "the big apple"]
      }
    ],
    "common_examples" : [
      {
        "text": "show me flights to New York City",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 32,
            "value": "NYC"
          }
        ]
      },
      {
        "text": "show me flights to nyc",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 22,
            "value": "nyc"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode(io_utils.DEFAULT_ENCODING))
        f.flush()
        td = training_data.load_data(f.name)
        assert td.entity_synonyms["New York City"] == "nyc"


def cmp_message_list(firsts, seconds):
    assert len(firsts) == len(seconds), "Message lists have unequal length"


def cmp_dict_list(firsts, seconds):
    if len(firsts) != len(seconds):
        return False

    for a in firsts:
        for idx, b in enumerate(seconds):
            if hash(a) == hash(b):
                del seconds[idx]
                break
        else:
            others = ", ".join([e.text for e in seconds])
            assert False, f"Failed to find message {a.text} in {others}"
    return not seconds


@pytest.mark.parametrize(
    "data_file,gold_standard_file,output_format,language",
    [
        (
            "data/examples/wit/demo-flights.json",
            "data/test/wit_converted_to_rasa.json",
            "json",
            None,
        ),
        (
            "data/examples/luis/demo-restaurants_v5.json",
            "data/test/luis_converted_to_rasa.json",
            "json",
            None,
        ),
        (
            "data/examples/dialogflow/",
            "data/test/dialogflow_en_converted_to_rasa.json",
            "json",
            "en",
        ),
        (
            "data/examples/dialogflow/",
            "data/test/dialogflow_es_converted_to_rasa.json",
            "json",
            "es",
        ),
        (
            "data/examples/rasa/demo-rasa.md",
            "data/test/md_converted_to_json.json",
            "json",
            None,
        ),
        (
            "data/examples/rasa/demo-rasa.json",
            "data/test/json_converted_to_md.md",
            "md",
            None,
        ),
        (
            "data/test/training_data_containing_special_chars.json",
            "data/test/json_with_special_chars_convered_to_md.md",
            "md",
            None,
        ),
    ],
)
def test_training_data_conversion(
    tmpdir, data_file, gold_standard_file, output_format, language
):
    out_path = tmpdir.join("rasa_nlu_data.json")
    convert_training_data(data_file, out_path.strpath, output_format, language)
    td = training_data.load_data(out_path.strpath, language)
    assert td.entity_examples != []
    assert td.intent_examples != []

    gold_standard = training_data.load_data(gold_standard_file, language)
    cmp_message_list(td.entity_examples, gold_standard.entity_examples)
    cmp_message_list(td.intent_examples, gold_standard.intent_examples)
    assert td.entity_synonyms == gold_standard.entity_synonyms

    # converting the converted file back to original
    # file format and performing the same tests
    rto_path = tmpdir.join("data_in_original_format.txt")
    convert_training_data(out_path.strpath, rto_path.strpath, "json", language)
    rto = training_data.load_data(rto_path.strpath, language)
    cmp_message_list(gold_standard.entity_examples, rto.entity_examples)
    cmp_message_list(gold_standard.intent_examples, rto.intent_examples)
    assert gold_standard.entity_synonyms == rto.entity_synonyms

    # If the above assert fails - this can be used
    # to dump to the file and diff using git
    # with io.open(gold_standard_file) as f:
    #     f.write(td.as_json(indent=2))


def test_url_data_format():
    data = """
    {
      "rasa_nlu_data": {
        "entity_synonyms": [
          {
            "value": "nyc",
            "synonyms": ["New York City", "nyc", "the big apple"]
          }
        ],
        "common_examples" : [
          {
            "text": "show me flights to New York City",
            "intent": "unk",
            "entities": [
              {
                "entity": "destination",
                "start": 19,
                "end": 32,
                "value": "NYC"
              }
            ]
          }
        ]
      }
    }"""
    fname = io_utils.create_temporary_file(
        data.encode(io_utils.DEFAULT_ENCODING),
        suffix="_tmp_training_data.json",
        mode="w+b",
    )
    data = io_utils.read_json_file(fname)
    assert data is not None
    validate_rasa_nlu_data(data)


def test_markdown_entity_regex():
    r = MarkdownReader()

    md = """
## intent:restaurant_search
- i'm looking for a place to eat
- i'm looking for a place in the [north](loc-direction) of town
- show me [chines](cuisine:chinese) restaurants
- show me [chines](22_ab-34*3.A:43er*+?df) restaurants
    """

    result = r.reads(md)

    assert len(result.training_examples) == 4
    first = result.training_examples[0]
    assert first.data == {"intent": "restaurant_search"}
    assert first.text == "i'm looking for a place to eat"

    second = result.training_examples[1]
    assert second.data == {
        "intent": "restaurant_search",
        "entities": [
            {"start": 31, "end": 36, "value": "north", "entity": "loc-direction"}
        ],
    }
    assert second.text == "i'm looking for a place in the north of town"

    third = result.training_examples[2]
    assert third.data == {
        "intent": "restaurant_search",
        "entities": [{"start": 8, "end": 14, "value": "chinese", "entity": "cuisine"}],
    }
    assert third.text == "show me chines restaurants"

    fourth = result.training_examples[3]
    assert fourth.data == {
        "intent": "restaurant_search",
        "entities": [
            {"start": 8, "end": 14, "value": "43er*+?df", "entity": "22_ab-34*3.A"}
        ],
    }
    assert fourth.text == "show me chines restaurants"


def test_get_file_format():
    fformat = get_file_format("data/examples/luis/demo-restaurants_v5.json")

    assert fformat == "json"

    fformat = get_file_format("data/examples")

    assert fformat == "json"

    fformat = get_file_format("examples/restaurantbot/data/nlu.md")

    assert fformat == "md"

    with pytest.raises(AttributeError):
        get_file_format("path-does-not-exists")

    with pytest.raises(AttributeError):
        get_file_format(None)


def test_guess_format_from_non_existing_file_path():
    assert guess_format("not existing path") == UNK


def test_load_data_from_non_existing_file():
    with pytest.raises(ValueError):
        load_data("some path")


def test_is_empty():
    assert TrainingData().is_empty()


def test_markdown_empty_section():
    data = training_data.load_data(
        "data/test/markdown_single_sections/empty_section.md"
    )
    assert data.regex_features == [{"name": "greet", "pattern": r"hey[^\s]*"}]

    assert not data.entity_synonyms
    assert len(data.lookup_tables) == 1
    assert data.lookup_tables[0]["name"] == "chinese"
    assert "Chinese" in data.lookup_tables[0]["elements"]
    assert "Chines" in data.lookup_tables[0]["elements"]


def test_markdown_not_existing_section():
    with pytest.raises(ValueError):
        training_data.load_data(
            "data/test/markdown_single_sections/not_existing_section.md"
        )


def test_section_value_with_delimiter():
    td_section_with_delimiter = training_data.load_data(
        "data/test/markdown_single_sections/section_with_delimiter.md"
    )
    assert td_section_with_delimiter.entity_synonyms == {"10:00 am": "10:00"}


def test_markdown_order():
    r = MarkdownReader()

    md = """## intent:z
- i'm looking for a place to eat
- i'm looking for a place in the [north](loc-direction) of town

## intent:a
- intent a
- also very important
"""

    training_data = r.reads(md)
    assert training_data.nlu_as_markdown() == md


def test_dump_nlu_with_responses():
    md = """## intent:greet
- hey
- howdy
- hey there
- hello
- hi
- good morning
- good evening
- dear sir

## intent:chitchat/ask_name
- What's your name?
- What can I call you?

## intent:chitchat/ask_weather
- How's the weather?
- Is it too hot outside?
"""

    r = MarkdownReader()
    nlu_data = r.reads(md)

    dumped = nlu_data.nlu_as_markdown()
    assert dumped == md


@pytest.mark.parametrize(
    "entity_extractor,expected_output",
    [
        (None, "- [test](word:random)"),
        ("", "- [test](word:random)"),
        ("random-extractor", "- [test](word:random)"),
        (CRFEntityExtractor.__name__, "- [test](word:random)"),
        (DucklingHTTPExtractor.__name__, "- test"),
        (SpacyEntityExtractor.__name__, "- test"),
        (MitieEntityExtractor.__name__, "- [test](word:random)"),
    ],
)
def test_dump_trainable_entities(
    entity_extractor: Optional[Text], expected_output: Text
):
    training_data_json = {
        "rasa_nlu_data": {
            "common_examples": [
                {
                    "text": "test",
                    "intent": "greet",
                    "entities": [
                        {"start": 0, "end": 4, "value": "random", "entity": "word"}
                    ],
                }
            ]
        }
    }
    if entity_extractor is not None:
        training_data_json["rasa_nlu_data"]["common_examples"][0]["entities"][0][
            "extractor"
        ] = entity_extractor

    training_data_object = RasaReader().read_from_json(training_data_json)
    md_dump = MarkdownWriter().dumps(training_data_object)
    assert md_dump.splitlines()[1] == expected_output
