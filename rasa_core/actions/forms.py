# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.actions.action import Action
from rasa_core.dispatcher import Button
from rasa_core.events import ConversationPaused, SlotSet

import logging
import random

logger = logging.getLogger(__name__)



class FormField(object):

    def validate(self, value):
        return value is not None


class EntityFormField(FormField):

    def __init__(self, entity_name, slot_name):
        self.entity_name = entity_name
        self.slot_name = slot_name

    def extract(self, tracker):
        value = None
        for e in tracker.latest_message.entities:
            if e["entity"] == self.entity_name:
                value = e["value"]
        if value:
            return [SlotSet(self.slot_name, value)]
        else:
            return []


class BooleanFormField(FormField):

    def __init__(self, slot_name, affirm_intent, deny_intent):
        self.slot_name = slot_name
        self.affirm_intent = affirm_intent
        self.deny_intent = deny_intent

    def extract(self, tracker):
        value = None
        intent = tracker.latest_message.intent["name"]
        if intent == self.affirm_intent:
            value = True
        elif intent == self.deny_intent:
            value = False
        return [SlotSet(self.slot_name, value)]



class FormAction(Action):

    REQUIRED_FIELDS = []
    RANDOMIZE = True

    def should_request_slot(self, tracker, slot_name, events):
        existing_val = tracker.get_slot(slot_name)
        pending = [e.key for e in events if e.key == slot_name]
        return existing_val is None and not slot_name in pending

    def get_requested_slot(self, tracker):
        requested_slot = tracker.get_slot("requested_slot")

        if requested_slot is None:
            return []
        else:
            try:
                required = self.REQUIRED_FIELDS[:]
                if self.RANDOMIZE:
                    random.shuffle(required)
                fields = [f for f in required if f.slot_name == requested_slot]
                return fields[0].extract(tracker)
            except:
                raise
                return []

    def ready_to_submit(self, tracker, events):
        return not any([
            self.should_request_slot(tracker, field.slot_name, events)
            for field in self.REQUIRED_FIELDS])


    def run(self, dispatcher, tracker, domain):

        events = self.get_requested_slot(tracker)

        if self.ready_to_submit(tracker, events):
            return self.submit(dispatcher, tracker, domain)

        for field in self.REQUIRED_FIELDS:
            if self.should_request_slot(tracker, field.slot_name, events):
                dispatcher.utter_template("utter_ask_{}".format(field.slot_name))

                events.append(SlotSet("requested_slot", field.slot_name))
                return events

        return self.submit(dispatcher, tracker, domain)


    def submit(self, dispatcher, tracker, domain):
        dispatcher.utter_message("done!")
        return []

