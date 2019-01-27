import logging
import matplotlib
import os
import pytest

from rasa_core import train, server, utils
from rasa_core.agent import Agent
from rasa_core.channels import CollectingOutputChannel, RestInput, channel
from rasa_core.dispatcher import Dispatcher
from rasa_core.domain import Domain
from rasa_core.interpreter import RegexInterpreter
from rasa_core.nlg import TemplatedNaturalLanguageGenerator
from rasa_core.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from rasa_core.policies.memoization import (
    Policy, MemoizationPolicy, AugmentedMemoizationPolicy)
from rasa_core.processor import MessageProcessor
from rasa_core.slots import Slot
from rasa_core.tracker_store import InMemoryTrackerStore
from rasa_core.trackers import DialogueStateTracker
from rasa_core.utils import zip_folder

matplotlib.use('Agg')

logging.basicConfig(level="DEBUG")

DEFAULT_DOMAIN_PATH = "data/test_domains/default_with_slots.yml"

DEFAULT_STORIES_FILE = "data/test_stories/stories_defaultdomain.md"

END_TO_END_STORY_FILE = "data/test_evaluations/end_to_end_story.md"

E2E_STORY_FILE_UNKNOWN_ENTITY = "data/test_evaluations/story_unknown_entity.md"

MOODBOT_MODEL_PATH = "examples/moodbot/models/dialogue"

DEFAULT_ENDPOINTS_FILE = "data/test_endpoints/example_endpoints.yml"


class CustomSlot(Slot):
    def as_feature(self):
        return [0.5]


class ExamplePolicy(Policy):

    def __init__(self, example_arg):
        pass


@pytest.fixture
def loop():
    from pytest_sanic.plugin import loop as sanic_loop
    l = next(sanic_loop())
    utils.enable_async_loop_debugging(l)
    return l


@pytest.fixture(scope="session")
def default_domain():
    return Domain.load(DEFAULT_DOMAIN_PATH)


@pytest.fixture(scope="session")
async def default_agent(default_domain):
    agent = Agent(default_domain,
                  policies=[MemoizationPolicy()],
                  interpreter=RegexInterpreter(),
                  tracker_store=InMemoryTrackerStore(default_domain))
    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    return agent


@pytest.fixture(scope="session")
def default_agent_path(default_agent, tmpdir_factory):
    path = tmpdir_factory.mktemp("agent").strpath
    default_agent.persist(path)
    return path


@pytest.fixture
def default_dispatcher_collecting(default_nlg):
    bot = CollectingOutputChannel()
    return Dispatcher("my-sender", bot, default_nlg)


@pytest.fixture
async def default_processor(default_domain, default_nlg):
    agent = Agent(default_domain,
                  SimplePolicyEnsemble([AugmentedMemoizationPolicy()]),
                  interpreter=RegexInterpreter())

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    tracker_store = InMemoryTrackerStore(default_domain)
    return MessageProcessor(agent.interpreter,
                            agent.policy_ensemble,
                            default_domain,
                            tracker_store,
                            default_nlg)


@pytest.fixture(scope="session")
async def trained_moodbot_path():
    await train.train_dialogue_model(
        domain_file="examples/moodbot/domain.yml",
        stories_file="examples/moodbot/data/stories.md",
        output_path=MOODBOT_MODEL_PATH,
        interpreter=RegexInterpreter(),
        policy_config='rasa_core/default_config.yml',
        kwargs=None
    )

    return MOODBOT_MODEL_PATH


@pytest.fixture(scope="session")
def zipped_moodbot_model():
    # train moodbot if necessary
    policy_file = os.path.join(MOODBOT_MODEL_PATH, 'policy_metadata.json')
    if not os.path.isfile(policy_file):
        trained_moodbot_path()

    zip_path = zip_folder(MOODBOT_MODEL_PATH)

    return zip_path


@pytest.fixture(scope="session")
def moodbot_domain():
    domain_path = os.path.join(MOODBOT_MODEL_PATH, 'domain.yml')
    return Domain.load(domain_path)


@pytest.fixture(scope="session")
def moodbot_metadata():
    return PolicyEnsemble.load_metadata(MOODBOT_MODEL_PATH)


@pytest.fixture
async def prepared_agent(tmpdir_factory):
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent("data/test_domains/default.yml",
                  policies=[AugmentedMemoizationPolicy(max_history=3)])

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    agent.persist(model_path)
    return agent


@pytest.fixture
async def core_server(prepared_agent):
    app = server.create_app(prepared_agent)
    channel.register([RestInput()],
                     app,
                     prepared_agent.handle_message,
                     "/webhooks/")
    return app


@pytest.fixture
async def core_server_secured(prepared_agent):
    app = server.create_app(prepared_agent,
                            auth_token="rasa",
                            jwt_secret="core")
    channel.register([RestInput()],
                     app,
                     prepared_agent.handle_message,
                     "/webhooks/")
    return app


@pytest.fixture
def default_nlg(default_domain):
    return TemplatedNaturalLanguageGenerator(default_domain.templates)


@pytest.fixture
def default_tracker(default_domain):
    import uuid
    uid = str(uuid.uuid1())
    return DialogueStateTracker(uid, default_domain.slots)
