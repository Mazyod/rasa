from typing import List, Text

import pytest

from rasa.core import training
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.constants import REQUESTED_SLOT, RULE_SNIPPET_ACTION_NAME
from rasa.core.domain import Domain
from rasa.core.events import (
    ActionExecuted,
    UserUttered,
    Form,
    SlotSet,
    ActionExecutionRejected,
)
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.generator import TrackerWithCachedStates

UTTER_GREET_ACTION = "utter_greet"
GREET_INTENT_NAME = "greet"
GREET_RULE = DialogueStateTracker.from_events(
    "bla",
    evts=[
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        # Greet is a FAQ here and gets triggered in any context
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ActionExecuted(UTTER_GREET_ACTION),
    ],
)
GREET_RULE.is_rule_tracker = True


def _form_submit_rule(
    domain: Domain, submit_action_name: Text, form_name: Text
) -> DialogueStateTracker:
    return TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            Form(form_name),
            # Any events in between
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            # Form runs and deactivates itself
            ActionExecuted(form_name),
            Form(None),
            SlotSet(REQUESTED_SLOT, None),
            ActionExecuted(submit_action_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )


def _form_activation_rule(
    domain: Domain, form_name: Text, activation_intent_name: Text
) -> DialogueStateTracker:
    return TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            # The intent `other_intent` activates the form
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": activation_intent_name}),
            ActionExecuted(form_name),
            Form(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )


def test_faq_rule():
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)
    new_conversation = DialogueStateTracker.from_events(
        "bla2", GREET_RULE.applied_events()[1:-1]
    )
    action_probabilities = policy.predict_action_probabilities(new_conversation, domain)

    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)


def assert_predicted_action(
    action_probabilities: List[float], domain: Domain, expected_action_name: Text
) -> None:
    assert max(action_probabilities) == 1
    index_of_predicted_action = action_probabilities.index(max(action_probabilities))
    prediction_action_name = domain.action_names[index_of_predicted_action]
    assert prediction_action_name == expected_action_name


async def test_predict_form_action_if_in_form():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
    intents:
    - {GREET_INTENT_NAME}
    actions:
    - {UTTER_GREET_ACTION}
    - some-action
    slots:
      {REQUESTED_SLOT}:
        type: unfeaturized
    forms:
    - {form_name}
"""
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an activate form
            Form(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User sends message as response to a requested slot
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ],
        slots=domain.slots,
    )

    # RulePolicy triggers form again
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, form_name)


async def test_predict_form_action_if_multiple_turns():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
    intents:
    - {GREET_INTENT_NAME}
    - {other_intent}
    actions:
    - {UTTER_GREET_ACTION}
    - some-action
    slots:
      {REQUESTED_SLOT}:
        type: unfeaturized
    forms:
    - {form_name}
"""
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an active form
            Form(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            # User responds to slot request
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            # Form validates input and requests another slot
            ActionExecuted(form_name),
            SlotSet(REQUESTED_SLOT, "some other"),
            # User responds to 2nd slot request
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": other_intent}),
        ],
        slots=domain.slots,
    )

    # RulePolicy triggers form again
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, form_name)


async def test_dont_predict_form_if_already_finished():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
    intents:
    - {GREET_INTENT_NAME}
    actions:
    - {UTTER_GREET_ACTION}
    - some-action
    slots:
      {REQUESTED_SLOT}:
        type: unfeaturized
    forms:
    - {form_name}
"""
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an activate form
            Form(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User sends message as response to a requested slot
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            # Form is happy and deactivates itself
            ActionExecuted(form_name),
            Form(None),
            SlotSet(REQUESTED_SLOT, None),
            # User sends another message. Form is already done. Shouldn't get triggered
            # again
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ],
        slots=domain.slots,
    )

    # RulePolicy triggers form again
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)


async def test_form_unhappy_path():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)

    unhappy_form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an active form
            Form(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            # User responds to slot request
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            # Form isn't happy with the answer and rejects execution
            Form(form_name),
            ActionExecutionRejected(form_name),
        ],
        slots=domain.slots,
    )

    # RulePolicy doesn't trigger form but FAQ
    action_probabilities = policy.predict_action_probabilities(
        unhappy_form_conversation, domain
    )

    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)


async def test_form_unhappy_path_triggering_form_again():
    form_name = "some_form"
    handle_rejection_action_name = "utter_handle_rejection"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - {handle_rejection_action_name}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    unhappy_rule = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            # We are in an active form
            Form(form_name),
            SlotSet(REQUESTED_SLOT, "bla"),
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            # When a user says "hi", and the form is unhappy, we want to run a specific
            # action
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(handle_rejection_action_name),
            ActionExecuted(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    policy = RulePolicy()
    policy.train([unhappy_rule], domain)

    # Check that RulePolicy predicts action to handle unhappy path
    conversation_events = [
        Form(form_name),
        SlotSet(REQUESTED_SLOT, "some value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
        Form(form_name),
        ActionExecutionRejected(form_name),
    ]

    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    assert_predicted_action(action_probabilities, domain, handle_rejection_action_name)

    # Check that RulePolicy triggers form again after handling unhappy path
    conversation_events.append(ActionExecuted(handle_rejection_action_name))
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    assert_predicted_action(action_probabilities, domain, form_name)


async def test_form_unhappy_path_without_rule():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        - {other_intent}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)

    conversation_events = [
        Form(form_name),
        SlotSet(REQUESTED_SLOT, "some value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": other_intent}),
        Form(form_name),
        ActionExecutionRejected(form_name),
    ]

    # Unhappy path is not handled. No rule matches. Let's hope ML fixes our problems 🤞
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )

    assert max(action_probabilities) == 0


async def test_form_activation_rule():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        - {other_intent}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    form_activation_rule = _form_activation_rule(domain, form_name, other_intent)
    policy = RulePolicy()
    policy.train([GREET_RULE, form_activation_rule], domain)

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": other_intent}),
    ]

    # RulePolicy correctly predicts the form action
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )

    assert_predicted_action(action_probabilities, domain, form_name)


async def test_failing_form_activation_due_to_no_rule():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        - {other_intent}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": other_intent}),
    ]

    # RulePolicy has no matching rule since no rule for form activation is given
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )

    assert max(action_probabilities) == 0


def test_form_submit_rule():
    form_name = "some_form"
    submit_action_name = "utter_submit"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        - {submit_action_name}
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    form_submit_rule = _form_submit_rule(domain, submit_action_name, form_name)

    policy = RulePolicy()
    policy.train([GREET_RULE, form_submit_rule], domain)

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # Form was activated
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            Form(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User responds and fills requested slot
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(form_name),
            # Form get's deactivated
            Form(None),
            SlotSet(REQUESTED_SLOT, None),
        ],
        slots=domain.slots,
    )

    # RulePolicy predicts action which handles submit
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, submit_action_name)


def test_immediate_submit():
    form_name = "some_form"
    submit_action_name = "utter_submit"
    entity = "some_entity"
    slot = "some_slot"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        - {submit_action_name}
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
          {slot}:
            type: unfeaturized
        forms:
        - {form_name}
        entities:
        - {entity}
    """
    )

    form_activation_rule = _form_activation_rule(domain, form_name, GREET_INTENT_NAME)
    form_submit_rule = _form_submit_rule(domain, submit_action_name, form_name)

    policy = RulePolicy()
    policy.train([GREET_RULE, form_activation_rule, form_submit_rule], domain)

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # Form was activated
            ActionExecuted(ACTION_LISTEN_NAME),
            # The same intent which activates the form also deactivates it
            UserUttered(
                "haha",
                {"name": GREET_INTENT_NAME},
                entities=[{"entity": entity, "value": "Bruce Wayne"}],
            ),
            SlotSet(slot, "Bruce"),
            ActionExecuted(form_name),
            SlotSet("bla", "bla"),
            Form(None),
            SlotSet(REQUESTED_SLOT, None),
        ],
        slots=domain.slots,
    )

    # RulePolicy predicts action which handles submit
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, submit_action_name)


@pytest.fixture(scope="session")
def trained_rule_policy_domain() -> Domain:
    return Domain.load("examples/rules/domain.yml")


@pytest.fixture(scope="session")
async def trained_rule_policy(trained_rule_policy_domain: Domain) -> RulePolicy:
    trackers = await training.load_data(
        "examples/rules/data/stories.md", trained_rule_policy_domain
    )

    rule_policy = RulePolicy()
    rule_policy.train(trackers, trained_rule_policy_domain)

    return rule_policy


async def test_rule_policy_slot_filling_from_text(
    trained_rule_policy: RulePolicy, trained_rule_policy_domain: Domain
):
    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            # User responds and fills requested slot
            UserUttered("/activate_q_form", {"name": "activate_q_form"}),
            ActionExecuted("loop_q_form"),
            Form("loop_q_form"),
            SlotSet(REQUESTED_SLOT, "some_slot"),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("/bla", {"name": GREET_INTENT_NAME}),
            ActionExecuted("loop_q_form"),
            SlotSet("some_slot", "/bla"),
            Form(None),
            SlotSet(REQUESTED_SLOT, None),
        ],
        slots=trained_rule_policy_domain.slots,
    )

    # RulePolicy predicts action which handles submit
    action_probabilities = trained_rule_policy.predict_action_probabilities(
        form_conversation, trained_rule_policy_domain
    )
    assert_predicted_action(
        action_probabilities, trained_rule_policy_domain, "utter_stop"
    )
