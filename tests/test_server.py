# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import signal
import uuid
from multiprocessing import Process

import pytest
from builtins import str
from freezegun import freeze_time
from pytest_localserver.http import WSGIServer

import rasa_core
from rasa_core import server, events
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.agent import Agent
from rasa_core.channels import UserMessage
from rasa_core.channels import CollectingOutputChannel
from rasa_core.events import (
    UserUttered, BotUttered, SlotSet, Event, ActionExecuted)
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.memoization import AugmentedMemoizationPolicy
from rasa_core.remote import RasaCoreClient
from tests.conftest import DEFAULT_STORIES_FILE

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


@pytest.fixture(scope="module")
def app(core_server):
    return core_server.test_client()


def test_root(app):
    response = app.get("http://dummy/")
    content = response.get_data(as_text=True)
    assert response.status_code == 200 and content.startswith("hello")


def test_version(app):
    response = app.get("http://dummy/version")
    content = response.get_json()
    assert response.status_code == 200
    assert content.get("version") == rasa_core.__version__


@freeze_time("2018-01-01")
def test_requesting_non_existent_tracker(app):
    response = app.get("http://dummy/conversations/madeupid/tracker")
    content = response.get_json()
    assert response.status_code == 200
    assert content["paused"] is False
    assert content["slots"] == {"location": None, "cuisine": None}
    assert content["sender_id"] == "madeupid"
    assert content["events"] == [{"event": "action",
                                  "name": "action_listen",
                                  "timestamp": 1514764800}]
    assert content["latest_message"] == {"text": None,
                                         "intent": {},
                                         "entities": []}


def test_respond(app):
    data = json.dumps({"query": "/greet"})
    response = app.post("http://dummy/conversations/myid/respond",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200
    assert content == [{'text': 'hey there!', 'recipient_id': 'myid'}]


@pytest.mark.parametrize("event", test_events)
def test_pushing_events(app, event):
    cid = str(uuid.uuid1())
    conversation = "http://dummy/conversations/{}".format(cid)
    data = json.dumps({"query": "/greet"})
    response = app.post("{}/respond".format(conversation),
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    data = json.dumps([event.as_dict()])
    response = app.post("{}/tracker/events".format(conversation),
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    tracker_response = app.get("http://dummy/conversations/{}/tracker"
                               "".format(cid))
    tracker = tracker_response.get_json()
    assert tracker is not None
    assert len(tracker.get("events")) == 6

    evt = tracker.get("events")[5]
    assert Event.from_parameters(evt) == event


def test_put_tracker(app):
    data = json.dumps([event.as_dict() for event in test_events])
    response = app.put("http://dummy/conversations/pushtracker/tracker",
                       data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200
    assert len(content["events"]) == len(test_events)
    assert content["sender_id"] == "pushtracker"

    tracker_response = app.get("http://dummy/conversations/pushtracker/tracker")
    tracker = tracker_response.get_json()
    assert tracker is not None
    evts = tracker.get("events")
    assert events.deserialise_events(evts) == test_events


def test_list_conversations(app):
    data = json.dumps({"query": "/greet"})
    response = app.post("http://dummy/conversations/myid/respond",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    response = app.get("http://dummy/conversations")
    content = response.get_json()
    assert response.status_code == 200

    assert len(content) > 0
    assert "myid" in content


def test_remote_status(http_app):
    client = RasaCoreClient(http_app, None)

    status = client.status()

    assert status.get("version") == rasa_core.__version__


def test_remote_clients(http_app):
    client = RasaCoreClient(http_app, None)

    cid = str(uuid.uuid1())
    client.respond("/greet", cid)

    clients = client.clients()

    assert cid in clients


def test_remote_append_events(http_app):
    client = RasaCoreClient(http_app, None)

    cid = str(uuid.uuid1())

    client.append_events_to_tracker(cid, test_events[:2])

    tracker = client.tracker_json(cid)

    evts = tracker.get("events")
    expected = [ActionExecuted(ACTION_LISTEN_NAME)] + test_events[:2]
    assert events.deserialise_events(evts) == expected
