DOCS_BASE_URL = "https://rasa.com/docs/rasa"
LEGACY_DOCS_BASE_URL = "https://legacy-docs-v1.rasa.com"
DOCS_URL_TRAINING_DATA = DOCS_BASE_URL + "/training-data-format"
DOCS_URL_TRAINING_DATA_NLU = DOCS_URL_TRAINING_DATA + "#nlu-training-data"
DOCS_URL_DOMAINS = DOCS_BASE_URL + "/domain"
DOCS_URL_SLOTS = DOCS_URL_DOMAINS + "#slots"
DOCS_URL_RESPONSES = DOCS_BASE_URL + "/responses"
DOCS_URL_STORIES = DOCS_BASE_URL + "/stories"
DOCS_URL_RULES = DOCS_BASE_URL + "/rules"
DOCS_URL_FORMS = DOCS_BASE_URL + "/forms"
DOCS_URL_PIPELINE = DOCS_BASE_URL + "/tuning-your-model"
DOCS_URL_POLICIES = DOCS_BASE_URL + "/policies"
DOCS_URL_TEST_STORIES = DOCS_BASE_URL + "/testing-your-assistant"
DOCS_URL_ACTIONS = DOCS_BASE_URL + "/actions"
DOCS_URL_CONNECTORS = DOCS_BASE_URL + "/connectors/"
DOCS_URL_EVENT_BROKERS = DOCS_BASE_URL + "/event-brokers"
DOCS_URL_PIKA_EVENT_BROKER = DOCS_URL_EVENT_BROKERS + "#pika-event-broker"
DOCS_URL_TRACKER_STORES = DOCS_BASE_URL + "/tracker-stores"
DOCS_URL_COMPONENTS = DOCS_BASE_URL + "/components"
DOCS_URL_MIGRATION_GUIDE = DOCS_BASE_URL + "/migration-guide"
DOCS_URL_TELEMETRY = DOCS_BASE_URL + "/telemetry/telemetry"
DOCS_BASE_URL_RASA_X = "https://rasa.com/docs/rasa-x"

INTENT_MESSAGE_PREFIX = "/"

PACKAGE_NAME = "rasa"
NEXT_MAJOR_VERSION_FOR_DEPRECATIONS = "3.0.0"

CONFIG_SCHEMA_FILE = "shared/nlu/training_data/schemas/config.yml"
RESPONSES_SCHEMA_FILE = "shared/nlu/training_data/schemas/responses.yml"
SCHEMA_EXTENSIONS_FILE = "shared/utils/pykwalify_extensions.py"
LATEST_TRAINING_DATA_FORMAT_VERSION = "2.0"

DOMAIN_SCHEMA_FILE = "utils/schemas/domain.yml"

DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES = 60
DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION = True

DEFAULT_NLU_FALLBACK_INTENT_NAME = "nlu_fallback"

DEFAULT_E2E_TESTS_PATH = "tests"
TEST_STORIES_FILE_PREFIX = "test_"

DEFAULT_LOG_LEVEL = "INFO"
ENV_LOG_LEVEL = "LOG_LEVEL"

DEFAULT_SENDER_ID = "default"
UTTER_PREFIX = "utter_"

CONFIG_AUTOCONFIGURABLE_KEYS_CORE = ["policies"]
CONFIG_AUTOCONFIGURABLE_KEYS_NLU = ["pipeline"]
CONFIG_AUTOCONFIGURABLE_KEYS = (
    CONFIG_AUTOCONFIGURABLE_KEYS_CORE + CONFIG_AUTOCONFIGURABLE_KEYS_NLU
)
CONFIG_KEYS_CORE = ["policies"]
CONFIG_KEYS_NLU = ["language", "pipeline"]
CONFIG_KEYS = CONFIG_KEYS_CORE + CONFIG_KEYS_NLU
CONFIG_MANDATORY_KEYS_CORE = []
CONFIG_MANDATORY_KEYS_NLU = ["language"]
CONFIG_MANDATORY_KEYS = CONFIG_MANDATORY_KEYS_CORE + CONFIG_MANDATORY_KEYS_NLU

# Constants for default Rasa Open Source project layout
DEFAULT_ENDPOINTS_PATH = "endpoints.yml"
DEFAULT_CREDENTIALS_PATH = "credentials.yml"
DEFAULT_CONFIG_PATH = "config.yml"
DEFAULT_DOMAIN_PATH = "domain.yml"
DEFAULT_ACTIONS_PATH = "actions"
DEFAULT_MODELS_PATH = "models"
DEFAULT_CONVERTED_DATA_PATH = "converted_data"
DEFAULT_DATA_PATH = "data"
DEFAULT_RESULTS_PATH = "results"
DEFAULT_NLU_RESULTS_PATH = "nlu_comparison_results"
DEFAULT_CORE_SUBDIRECTORY_NAME = "core"
DEFAULT_NLU_SUBDIRECTORY_NAME = "nlu"
