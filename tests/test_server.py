# -*- coding: utf-8 -*-
import io
import json
import pytest
import uuid
from freezegun import freeze_time

import rasa_core
from rasa_core import events, constants
from rasa_core.events import (
    UserUttered, BotUttered, SlotSet, Event)
from tests.conftest import DEFAULT_STORIES_FILE, END_TO_END_STORY_FILE

# a couple of event instances that we can use for testing
test_events = [
    Event.from_parameters({"event": UserUttered.type_name,
                           "text": "/goodbye",
                           "parse_data": {
                               "intent": {
                                   "confidence": 1.0, "name": "greet"},
                               "entities": []}
                           }),
    BotUttered("Welcome!", {"test": True}),
    SlotSet("cuisine", 34),
    SlotSet("cuisine", "34"),
    SlotSet("location", None),
    SlotSet("location", [34, "34", None]),
]


@pytest.fixture
def app(core_server):
    return core_server.test_client


@pytest.fixture
def secured_app(core_server_secured):
    return core_server_secured.test_client


def test_root(app):
    _, response = app.get("/")
    content = response.text
    assert response.status == 200
    assert content.startswith("hello")


def test_root_secured(secured_app):
    _, response = secured_app.get("/")
    content = response.text
    assert response.status == 200 and content.startswith("hello")


def test_version(app):
    _, response = app.get("/version")
    content = response.json
    assert response.status == 200
    assert content.get("version") == rasa_core.__version__
    assert (content.get("minimum_compatible_version") ==
            constants.MINIMUM_COMPATIBLE_VERSION)


def test_status(app):
    _, response = app.get("/status")
    content = response.json
    assert response.status == 200
    assert content.get("is_ready")
    assert content.get("model_fingerprint") is not None


@freeze_time("2018-01-01")
def test_requesting_non_existent_tracker(app):
    _, response = app.get("/conversations/madeupid/tracker")
    content = response.json
    assert response.status == 200
    assert content["paused"] is False
    assert content["slots"] == {"location": None, "cuisine": None}
    assert content["sender_id"] == "madeupid"
    assert content["events"] == [{"event": "action",
                                  "name": "action_listen",
                                  "policy": None,
                                  "confidence": None,
                                  "timestamp": 1514764800}]
    assert content["latest_message"] == {"text": None,
                                         "intent": {},
                                         "entities": []}


def test_respond(app):
    data = json.dumps({"query": "/greet"})
    _, response = app.post("/conversations/myid/respond",
                           data=data,
                           headers={"Content-Type": "application/json"})
    content = response.json
    assert response.status == 200
    assert content == [{'text': 'hey there!', 'recipient_id': 'myid'}]


@pytest.mark.parametrize("event", test_events)
def test_pushing_event(app, event):
    cid = str(uuid.uuid1())
    conversation = "/conversations/{}".format(cid)
    data = json.dumps({"query": "/greet"})
    _, response = app.post("{}/respond".format(conversation),
                           data=data,
                           headers={"Content-Type": "application/json"})
    assert response.json is not None
    assert response.status == 200

    data = json.dumps(event.as_dict())
    _, response = app.post("{}/tracker/events".format(conversation),
                           data=data,
                           headers={"Content-Type": "application/json"})
    assert (response.json is not None)
    assert response.status == 200

    _, tracker_response = app.get("/conversations/{}/tracker"
                                  "".format(cid))
    tracker = tracker_response.json
    assert tracker is not None
    assert len(tracker.get("events")) == 6

    evt = tracker.get("events")[5]
    assert Event.from_parameters(evt) == event


def test_put_tracker(app):
    data = json.dumps([event.as_dict() for event in test_events])
    _, response = app.put(
        "/conversations/pushtracker/tracker/events",
        data=data, headers={"Content-Type": "application/json"})
    content = response.json
    assert response.status == 200
    assert len(content["events"]) == len(test_events)
    assert content["sender_id"] == "pushtracker"

    _, tracker_response = app.get(
        "/conversations/pushtracker/tracker")
    tracker = tracker_response.json
    assert tracker is not None
    evts = tracker.get("events")
    assert events.deserialise_events(evts) == test_events


def test_list_conversations(app):
    data = json.dumps({"query": "/greet"})
    _, response = app.post("/conversations/myid/respond",
                           data=data,
                           headers={"Content-Type": "application/json"})
    assert response.json is not None
    assert response.status == 200

    _, response = app.get("/conversations")
    content = response.json
    assert response.status == 200

    assert len(content) > 0
    assert "myid" in content


def test_evaluate(app):
    with io.open(DEFAULT_STORIES_FILE, 'r') as f:
        stories = f.read()
    _, response = app.post('/evaluate',
                           data=stories)
    assert response.status == 200
    js = response.json
    assert set(js.keys()) == {"report",
                              "precision",
                              "f1",
                              "accuracy",
                              "actions",
                              "in_training_data_fraction",
                              "is_end_to_end_evaluation"}
    assert not js["is_end_to_end_evaluation"]
    assert set(js["actions"][0].keys()) == {
        "action",
        "predicted",
        "confidence",
        "policy"}


def test_end_to_end_evaluation(app):
    with io.open(END_TO_END_STORY_FILE, 'r') as f:
        stories = f.read()
    _, response = app.post('/evaluate?e2e=true',
                           data=stories)
    assert response.status == 200
    js = response.json
    assert set(js.keys()) == {"report",
                              "precision",
                              "f1",
                              "accuracy",
                              "actions",
                              "in_training_data_fraction",
                              "is_end_to_end_evaluation"}
    assert js["is_end_to_end_evaluation"]
    assert set(js["actions"][0].keys()) == {"action",
                                            "predicted",
                                            "confidence",
                                            "policy"}


def test_list_conversations_with_jwt(secured_app):
    # token generated with secret "core" and algorithm HS256
    # on https://jwt.io/

    # {"user": {"username": "testadmin", "role": "admin"}}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                         "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdGFkbWluIiwic"
                         "m9sZSI6ImFkbWluIn19.NAQr0kbtSrY7d28XTqRzawq2u"
                         "QRre7IWTuIDrCn5AIw"
    }
    _, response = secured_app.get("/conversations",
                                  headers=jwt_header)
    assert response.status == 200

    # {"user": {"username": "testuser", "role": "user"}}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                         "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdHVzZXIiLCJyb"
                         "2xlIjoidXNlciJ9fQ.JnMTLYd56qut2w9h7hRQlDm1n3l"
                         "HJHOxxC_w7TtwCrs"
    }
    _, response = secured_app.get("/conversations",
                                  headers=jwt_header)
    assert response.status == 403


def test_get_tracker_with_jwt(secured_app):
    # token generated with secret "core" and algorithm HS256
    # on https://jwt.io/

    # {"user": {"username": "testadmin", "role": "admin"}}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                         "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdGFkbWluIiwic"
                         "m9sZSI6ImFkbWluIn19.NAQr0kbtSrY7d28XTqRzawq2u"
                         "QRre7IWTuIDrCn5AIw"
    }
    _, response = secured_app.get("/conversations/testadmin/tracker",
                                  headers=jwt_header)
    assert response.status == 200

    _, response = secured_app.get("/conversations/testuser/tracker",
                                  headers=jwt_header)
    assert response.status == 200

    # {"user": {"username": "testuser", "role": "user"}}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                         "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdHVzZXIiLCJyb"
                         "2xlIjoidXNlciJ9fQ.JnMTLYd56qut2w9h7hRQlDm1n3l"
                         "HJHOxxC_w7TtwCrs"
    }
    print(json.dumps(jwt_header))
    _, response = secured_app.get("/conversations/testadmin/tracker",
                                  headers=jwt_header)
    assert response.status == 403

    _, response = secured_app.get("/conversations/testuser/tracker",
                                  headers=jwt_header)
    assert response.status == 200


def test_list_conversations_with_token(secured_app):
    _, response = secured_app.get("/conversations?token=rasa")
    assert response.status == 200


def test_list_conversations_with_wrong_token(secured_app):
    _, response = secured_app.get("/conversations?token=Rasa")
    assert response.status == 401


def test_list_conversations_without_auth(secured_app):
    _, response = secured_app.get("/conversations")
    assert response.status == 401


def test_list_conversations_with_wrong_jwt(secured_app):
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                         "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIi"
                         "wiaWF0IjoxNTE2MjM5MDIyfQ.qdrr2_a7Sd80gmCWjnDomO"
                         "Gl8eZFVfKXA6jhncgRn-I"
    }
    _, response = secured_app.get("/conversations",
                                  headers=jwt_header)
    assert response.status == 401


def test_story_export(app):
    data = json.dumps({"query": "/greet"})
    _, response = app.post("/conversations/mynewid/respond",
                           data=data,
                           headers={"Content-Type": "application/json"})
    assert response.status == 200
    _, response = app.get("/conversations/mynewid/story")
    assert response.status == 200
    story_lines = response.text.strip().split('\n')
    assert story_lines == ["## mynewid",
                           "* greet: /greet",
                           "    - utter_greet"]
