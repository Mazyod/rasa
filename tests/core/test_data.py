import json
import os
import shutil
import tempfile

import pytest

import rasa.data as data
from tests.core.conftest import DEFAULT_STORIES_FILE, DEFAULT_NLU_DATA
from rasa.nlu.training_data import load_data


def test_get_core_directory(project):
    data_dir = os.path.join(project, "data")
    core_directory = data.get_core_directory([data_dir])
    stories = os.listdir(core_directory)

    assert len(stories) == 1
    assert stories[0].endswith("stories.md")


def test_get_nlu_directory(project):
    data_dir = os.path.join(project, "data")
    nlu_directory = data.get_nlu_directory([data_dir])

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 1
    assert nlu_files[0].endswith("nlu.md")


def test_get_nlu_file(project):
    data_file = os.path.join(project, "data/nlu.md")
    nlu_directory = data.get_nlu_directory(data_file)

    nlu_files = os.listdir(nlu_directory)

    original = load_data(data_file)
    copied = load_data(nlu_directory)

    assert nlu_files[0].endswith("nlu.md")
    assert original.intent_examples == copied.intent_examples


def test_get_core_nlu_directories(project):
    data_dir = os.path.join(project, "data")
    core_directory, nlu_directory = data.get_core_nlu_directories([data_dir])

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 1
    assert nlu_files[0].endswith("nlu.md")

    stories = os.listdir(core_directory)

    assert len(stories) == 1
    assert stories[0].endswith("stories.md")


def test_get_core_nlu_directories_with_none():
    directories = data.get_core_nlu_directories(None)

    assert all([directory for directory in directories])
    assert all([not os.listdir(directory) for directory in directories])


def test_same_file_names_get_resolved(tmpdir):
    # makes sure the resolution properly handles if there are two files with
    # with the same name in different directories

    tmpdir.join("one").mkdir()
    tmpdir.join("two").mkdir()
    data_dir_one = os.path.join(tmpdir.join("one").join("stories.md").strpath)
    data_dir_two = os.path.join(tmpdir.join("two").join("stories.md").strpath)
    shutil.copy2(DEFAULT_STORIES_FILE, data_dir_one)
    shutil.copy2(DEFAULT_STORIES_FILE, data_dir_two)

    nlu_dir_one = os.path.join(tmpdir.join("one").join("nlu.md").strpath)
    nlu_dir_two = os.path.join(tmpdir.join("two").join("nlu.md").strpath)
    shutil.copy2(DEFAULT_NLU_DATA, nlu_dir_one)
    shutil.copy2(DEFAULT_NLU_DATA, nlu_dir_two)

    core_directory, nlu_directory = data.get_core_nlu_directories([tmpdir.strpath])

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 2
    assert all([f.endswith("nlu.md") for f in nlu_files])

    stories = os.listdir(core_directory)

    assert len(stories) == 2
    assert all([f.endswith("stories.md") for f in stories])


@pytest.mark.parametrize(
    "line",
    [
        "##intent:aintent",
        "##synonym: synonym",
        "##regex:a_regex",
        " ##lookup:additional",
    ],
)
def test_contains_nlu_pattern(line):
    assert data._contains_nlu_pattern(line)


def test_is_nlu_file_with_json():
    test = {
        "rasa_nlu_data": {
            "lookup_tables": [
                {"name": "plates", "elements": ["beans", "rice", "tacos", "cheese"]}
            ]
        }
    }

    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")
    with open(file, "w") as f:
        f.write(json.dumps(test))

    assert data._is_nlu_file(file)


def test_is_not_nlu_file_with_json():
    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")
    with open(file, "w") as f:
        f.write('{"test": "a"}')

    assert not data._is_nlu_file(file)


@pytest.mark.parametrize("line", ["- example", "## story intent 1 + two##slots* entry"])
def test_not_contains_nlu_pattern(line):
    assert not data._contains_nlu_pattern(line)
