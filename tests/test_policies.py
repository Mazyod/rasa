from typing import Text, List

from rasa_core.actions.action import ACTION_LISTEN_NAME

from rasa_core import training

from unittest.mock import patch
import numpy as np
import pytest

from rasa_core.channels import UserMessage
from rasa_core.domain import Domain
from rasa_core.policies import TwoStageFallbackPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import (
    MemoizationPolicy, AugmentedMemoizationPolicy)
from rasa_core.policies.sklearn_policy import SklearnPolicy
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.policies.embedding_policy import EmbeddingPolicy
from rasa_core.policies.form_policy import FormPolicy
from rasa_core.trackers import DialogueStateTracker
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE
from rasa_core.featurizers import (
    MaxHistoryTrackerFeaturizer,
    BinarySingleStateFeaturizer, FullDialogueTrackerFeaturizer)
from rasa_core.events import ActionExecuted, UserUttered, Event
from tests.utilities import read_dialogue_file


def train_trackers(domain):
    trackers = training.load_data(
        DEFAULT_STORIES_FILE,
        domain
    )
    return trackers


# We are going to use class style testing here since unfortunately pytest
# doesn't support using fixtures as arguments to its own parameterize yet
# (hence, we can't train a policy, declare it as a fixture and use the different
# fixtures of the different policies for the functional tests). Therefore, we
# are going to reverse this and train the policy within a class and collect the
# tests in a base class.
class PolicyTestCollection(object):
    """Tests every policy needs to fulfill.

    Each policy can declare further tests on its own."""

    max_history = 3  # this is the amount of history we test on

    def create_policy(self, featurizer):
        raise NotImplementedError

    @pytest.fixture(scope="module")
    def featurizer(self):
        featurizer = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                                 max_history=self.max_history)
        return featurizer

    @pytest.fixture(scope="module")
    def trained_policy(self, featurizer):
        default_domain = Domain.load(DEFAULT_DOMAIN_PATH)
        policy = self.create_policy(featurizer)
        training_trackers = train_trackers(default_domain)
        policy.train(training_trackers, default_domain)
        return policy

    def test_persist_and_load(self, trained_policy, default_domain, tmpdir):
        trained_policy.persist(tmpdir.strpath)
        loaded = trained_policy.__class__.load(tmpdir.strpath)
        trackers = train_trackers(default_domain)

        for tracker in trackers:
            predicted_probabilities = loaded.predict_action_probabilities(
                tracker, default_domain)
            actual_probabilities = trained_policy.predict_action_probabilities(
                tracker, default_domain)
            assert predicted_probabilities == actual_probabilities

    def test_prediction_on_empty_tracker(self, trained_policy, default_domain):
        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                       default_domain.slots)
        probabilities = trained_policy.predict_action_probabilities(
            tracker, default_domain)
        assert len(probabilities) == default_domain.num_actions
        assert max(probabilities) <= 1.0
        assert min(probabilities) >= 0.0

    def test_persist_and_load_empty_policy(self, tmpdir):
        empty_policy = self.create_policy(None)
        empty_policy.persist(tmpdir.strpath)
        loaded = empty_policy.__class__.load(tmpdir.strpath)
        assert loaded is not None


class TestKerasPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        p = KerasPolicy(featurizer)
        return p


class TestFallbackPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        p = FallbackPolicy()
        return p

    @pytest.mark.parametrize(
        "nlu_confidence, prev_action_is_fallback, should_fallback",
        [
            (0.1, True, False),
            (0.1, False, True),
            (0.9, True, False),
            (0.9, False, False),
        ])
    def test_something(self,
                       trained_policy,
                       nlu_confidence,
                       prev_action_is_fallback,
                       should_fallback):
        last_action_name = trained_policy.fallback_action_name if \
            prev_action_is_fallback else 'not_fallback'
        assert trained_policy.should_fallback(
            nlu_confidence, last_action_name) is should_fallback


class TestMemoizationPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        max_history = None
        if isinstance(featurizer, MaxHistoryTrackerFeaturizer):
            max_history = featurizer.max_history
        p = MemoizationPolicy(max_history=max_history)
        return p

    def test_memorise(self, trained_policy, default_domain):
        trackers = train_trackers(default_domain)
        trained_policy.train(trackers, default_domain)

        (all_states, all_actions) = \
            trained_policy.featurizer.training_states_and_actions(
                trackers, default_domain)

        for tracker, states, actions in zip(trackers, all_states, all_actions):
            recalled = trained_policy.recall(states, tracker, default_domain)
            assert recalled == default_domain.index_for_action(actions[0])

        nums = np.random.randn(default_domain.num_states)
        random_states = [{f: num
                          for f, num in
                          zip(default_domain.input_states, nums)}]
        assert trained_policy._recall_states(random_states) is None

    def test_memorise_with_nlu(self, trained_policy, default_domain):
        filename = "data/test_dialogues/nlu_dialogue.json"
        dialogue = read_dialogue_file(filename)

        tracker = DialogueStateTracker(dialogue.name, default_domain.slots)
        tracker.recreate_from_dialogue(dialogue)
        states = trained_policy.featurizer.prediction_states([tracker],
                                                             default_domain)[0]

        recalled = trained_policy.recall(states, tracker, default_domain)
        assert recalled is not None


class TestAugmentedMemoizationPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        max_history = None
        if isinstance(featurizer, MaxHistoryTrackerFeaturizer):
            max_history = featurizer.max_history
        p = AugmentedMemoizationPolicy(max_history=max_history)
        return p


class TestSklearnPolicy(PolicyTestCollection):

    def create_policy(self, featurizer, **kwargs):
        p = SklearnPolicy(featurizer, **kwargs)
        return p

    @pytest.yield_fixture
    def mock_search(self):
        with patch('rasa_core.policies.sklearn_policy.GridSearchCV') as gs:
            gs.best_estimator_ = 'mockmodel'
            gs.best_score_ = 0.123
            gs.return_value = gs  # for __init__
            yield gs

    @pytest.fixture(scope='module')
    def default_domain(self):
        return Domain.load(DEFAULT_DOMAIN_PATH)

    @pytest.fixture
    def tracker(self, default_domain):
        return DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                    default_domain.slots)

    @pytest.fixture(scope='module')
    def trackers(self, default_domain):
        return train_trackers(default_domain)

    def test_cv_none_does_not_trigger_search(self,
                                             mock_search,
                                             default_domain,
                                             trackers,
                                             featurizer):
        policy = self.create_policy(featurizer=featurizer, cv=None)
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count == 0
        assert policy.model != 'mockmodel'

    def test_cv_not_none_param_grid_none_triggers_search_without_params(
            self, mock_search, default_domain, trackers, featurizer):

        policy = self.create_policy(featurizer=featurizer, cv=3)
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]['cv'] == 3
        assert mock_search.call_args_list[0][1]['param_grid'] == {}
        assert policy.model == 'mockmodel'

    def test_cv_not_none_param_grid_none_triggers_search_with_params(
            self, mock_search, default_domain, trackers, featurizer):
        param_grid = {'n_estimators': 50}
        policy = self.create_policy(
            featurizer=featurizer,
            cv=3,
            param_grid=param_grid,
        )
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]['cv'] == 3
        assert mock_search.call_args_list[0][1]['param_grid'] == param_grid
        assert policy.model == 'mockmodel'

    def test_missing_classes_filled_correctly(
            self, default_domain, trackers, tracker, featurizer):
        # Pretend that a couple of classes are missing and check that
        # those classes are predicted as 0, while the other class
        # probabilities are predicted normally.
        policy = self.create_policy(featurizer=featurizer, cv=None)

        classes = [1, 3]
        new_trackers = []
        for tr in trackers:
            new_tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                               default_domain.slots)
            for e in tr.applied_events():
                if isinstance(e, ActionExecuted):
                    new_action = default_domain.action_for_index(
                        np.random.choice(classes),
                        action_endpoint=None).name()
                    new_tracker.update(ActionExecuted(new_action))
                else:
                    new_tracker.update(e)

            new_trackers.append(new_tracker)

        policy.train(new_trackers, domain=default_domain)
        predicted_probabilities = policy.predict_action_probabilities(
            tracker, default_domain)

        assert len(predicted_probabilities) == default_domain.num_actions
        assert np.allclose(sum(predicted_probabilities), 1.0)
        for i, prob in enumerate(predicted_probabilities):
            if i in classes:
                assert prob >= 0.0
            else:
                assert prob == 0.0

    def test_train_kwargs_are_set_on_model(
            self, default_domain, trackers, featurizer):
        policy = self.create_policy(featurizer=featurizer, cv=None)
        policy.train(trackers, domain=default_domain, C=123)
        assert policy.model.C == 123

    def test_train_with_shuffle_false(
            self, default_domain, trackers, featurizer):
        policy = self.create_policy(featurizer=featurizer, shuffle=False)
        # does not raise
        policy.train(trackers, domain=default_domain)


class TestEmbeddingPolicyNoAttention(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy()
        return p

    @pytest.fixture(scope="module")
    def trained_policy(self, featurizer):
        default_domain = Domain.load(DEFAULT_DOMAIN_PATH)
        policy = self.create_policy(featurizer)
        training_trackers = train_trackers(default_domain)
        policy.train(training_trackers, default_domain,
                     attn_before_rnn=False,
                     attn_after_rnn=False)
        return policy


class TestEmbeddingPolicyAttentionBeforeRNN(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy()
        return p

    @pytest.fixture(scope="module")
    def trained_policy(self, featurizer):
        default_domain = Domain.load(DEFAULT_DOMAIN_PATH)
        policy = self.create_policy(featurizer)
        training_trackers = train_trackers(default_domain)
        policy.train(training_trackers, default_domain,
                     attn_before_rnn=True,
                     attn_after_rnn=False)
        return policy


class TestEmbeddingPolicyAttentionAfterRNN(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy()
        return p

    @pytest.fixture(scope="module")
    def trained_policy(self, featurizer):
        default_domain = Domain.load(DEFAULT_DOMAIN_PATH)
        policy = self.create_policy(featurizer)
        training_trackers = train_trackers(default_domain)
        policy.train(training_trackers, default_domain,
                     attn_before_rnn=False,
                     attn_after_rnn=True)
        return policy


class TestEmbeddingPolicyAttentionBoth(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy()
        return p

    @pytest.fixture(scope="module")
    def trained_policy(self, featurizer):
        default_domain = Domain.load(DEFAULT_DOMAIN_PATH)
        policy = self.create_policy(featurizer)
        training_trackers = train_trackers(default_domain)
        policy.train(training_trackers, default_domain,
                     attn_before_rnn=True,
                     attn_after_rnn=True)
        return policy


class TestFormPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        p = FormPolicy()
        return p

    def test_memorise(self, trained_policy, default_domain):
        domain = Domain.load('data/test_domains/form.yml')
        trackers = training.load_data('data/test_stories/stories_form.md',
                                      domain)
        trained_policy.train(trackers, domain)

        (all_states, all_actions) = \
            trained_policy.featurizer.training_states_and_actions(
                trackers, domain)

        for tracker, states, actions in zip(trackers, all_states, all_actions):
            for state in states:
                if state is not None:
                    # check that 'form: inform' was ignored
                    assert 'intent_inform' not in state.keys()
            recalled = trained_policy.recall(states, tracker, domain)
            active_form = trained_policy._get_active_form_name(states[-1])

            if states[0] is not None and states[-1] is not None:
                # explicitly set intents and actions before listen after
                # which FormPolicy should not predict a form action and
                # should add FormValidation(False) event
                is_no_validation = (
                    ('prev_some_form' in states[0].keys() and
                     'intent_default' in states[-1].keys()) or
                    ('prev_some_form' in states[0].keys() and
                     'intent_stop' in states[-1].keys()) or
                    ('prev_utter_ask_continue' in states[0].keys() and
                     'intent_affirm' in states[-1].keys()) or
                    ('prev_utter_ask_continue' in states[0].keys() and
                     'intent_deny' in states[-1].keys())
                )
            else:
                is_no_validation = False

            if 'intent_start_form' in states[-1]:
                # explicitly check that intent that starts the form
                # is not memorized as non validation intent
                assert recalled is None
            elif is_no_validation:
                assert recalled == active_form
            else:
                assert recalled is None

        nums = np.random.randn(domain.num_states)
        random_states = [{f: num
                          for f, num in
                          zip(domain.input_states, nums)}]
        assert trained_policy.recall(random_states, None, domain) is None


def user_uttered(text: Text, confidence: float) -> UserUttered:
    parse_data = {'intent': {'name': text, 'confidence': confidence}}
    return UserUttered(text='Random', intent=text, parse_data=parse_data)


def get_tracker(events: List[Event]) -> DialogueStateTracker:
    return DialogueStateTracker.from_events("sender", events, [], 10)


class TestTwoStageFallbackPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        p = TwoStageFallbackPolicy()
        return p

    @pytest.fixture(scope="module")
    def domain(self):
        content = """
        actions:
          - action_ask_confirmation
          - action_ask_clarification
          - action_default_fallback
          - utter_hello

        intents:
          - greet
          - bye
          - confirm
          - deny
        """
        return Domain.from_yaml(content)

    def test_ask_confirmation(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("Hi", 0.2)]

        tracker = get_tracker(events)

        scores = trained_policy.predict_action_probabilities(tracker, domain)
        index = scores.index(max(scores))
        assert domain.action_names[index] == 'action_ask_confirmation'

    def test_confirmation(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("confirm", 1)]

        tracker = get_tracker(events)
        trained_policy.predict_action_probabilities(tracker, domain)

        assert 'greet' == tracker.latest_message.parse_data['intent']['name']

    def test_deny(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("deny", 1)]

        tracker = get_tracker(events)

        assert tracker.latest_message.parse_data['intent']['name'] == 'deny'

        scores = trained_policy.predict_action_probabilities(tracker, domain)
        index = scores.index(max(scores))
        assert domain.action_names[index] == 'action_ask_clarification'

    def test_successful_clarification(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("deny", 1),
                  ActionExecuted('action_ask_clarification'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("bye", 1),
                  ]

        tracker = get_tracker(events)
        trained_policy.predict_action_probabilities(tracker, domain)

        assert 'bye' == tracker.latest_message.parse_data['intent']['name']

    def test_confirm_clarification(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("deny", 1),
                  ActionExecuted('action_ask_clarification'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ]

        tracker = get_tracker(events)

        scores = trained_policy.predict_action_probabilities(tracker, domain)
        index = scores.index(max(scores))
        assert domain.action_names[index] == 'action_ask_confirmation'

    def test_confirmed_clarification(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("deny", 1),
                  ActionExecuted('action_ask_clarification'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("bye", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("confirm", 1)
                  ]

        tracker = get_tracker(events)
        trained_policy.predict_action_probabilities(tracker, domain)

        assert 'bye' == tracker.latest_message.parse_data['intent']['name']

    def test_denied_clarification_confirmation(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("deny", 1),
                  ActionExecuted('action_ask_clarification'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("bye", 0.2),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("deny", 1)
                  ]

        tracker = get_tracker(events)
        scores = trained_policy.predict_action_probabilities(tracker, domain)
        index = scores.index(max(scores))
        assert domain.action_names[index] == 'action_default_fallback'

    def test_clarification_instead_confirmation(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("bye", 1),
                  ]

        tracker = get_tracker(events)
        trained_policy.predict_action_probabilities(tracker, domain)

        assert 'bye' == tracker.latest_message.parse_data['intent']['name']

    def test_unknown_instead_confirmation(self, trained_policy, domain):
        events = [ActionExecuted(action_name=ACTION_LISTEN_NAME),
                  user_uttered("greet", 0.2),
                  ActionExecuted('action_ask_confirmation'),
                  ActionExecuted(ACTION_LISTEN_NAME),
                  user_uttered("bye", 0.2),
                  ]

        tracker = get_tracker(events)
        scores = trained_policy.predict_action_probabilities(tracker, domain)
        index = scores.index(max(scores))
        assert domain.action_names[index] == 'action_default_fallback'
