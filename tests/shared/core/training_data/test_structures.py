import rasa.core
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SessionStarted, SlotSet, UserUttered, ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import Story

domain = Domain.load("examples/moodbot/domain.yml")


def test_session_start_is_not_serialised(default_domain: Domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    # add SlotSet event
    tracker.update(SlotSet("slot", "value"))

    # add the two SessionStarted events and a user event
    tracker.update(ActionExecuted(ACTION_SESSION_START_NAME))
    tracker.update(SessionStarted())
    tracker.update(UserUttered("say something"))

    # make sure session start is not serialised
    story = Story.from_events(tracker.events, "some-story01")

    expected = """## some-story01
    - slot{"slot": "value"}
* say something
"""

    assert story.as_story_string(flat=True) == expected


def test_cap_length():
    assert (
        rasa.shared.core.training_data.structures._cap_length("mystring", 6) == "mys..."
    )


def test_cap_length_without_ellipsis():
    assert (
        rasa.shared.core.training_data.structures._cap_length(
            "mystring", 3, append_ellipsis=False
        )
        == "mys"
    )


def test_cap_length_with_short_string():
    assert rasa.shared.core.training_data.structures._cap_length("my", 3) == "my"
