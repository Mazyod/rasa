import os
from typing import Callable, Any, Tuple
from _pytest.pytester import RunResult


def test_init(run: Callable[[Tuple[Any]], RunResult]) -> None:
    run("init", "--no-prompt", "--quiet")

    assert os.path.exists("actions.py")
    assert os.path.exists("domain.yml")
    assert os.path.exists("config.yml")
    assert os.path.exists("credentials.yml")
    assert os.path.exists("endpoints.yml")
    assert os.path.exists("models")
    assert os.path.exists("data/nlu.md")
    assert os.path.exists("data/stories.md")


def test_init_help(run: Callable[[Tuple[Any]], RunResult]) -> None:
    output = run("init", "--help")

    assert (
        output.outlines[0] == "usage: rasa init [-h] [-v] [-vv] [--quiet] [--no-prompt]"
    )
