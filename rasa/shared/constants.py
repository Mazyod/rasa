DOCS_BASE_URL = "https://rasa.com/docs/rasa"
DOCS_URL_TRAINING_DATA_NLU = DOCS_BASE_URL + "/nlu/training-data-format/"
DOCS_URL_DOMAINS = DOCS_BASE_URL + "/core/domains/"

INTENT_MESSAGE_PREFIX = "/"

PACKAGE_NAME = "rasa"

CONFIG_SCHEMA_FILE = "shared/nlu/training_data/schemas/config.yml"
RESPONSES_SCHEMA_FILE = "shared/nlu/training_data/schemas/responses.yml"
SCHEMA_EXTENSIONS_FILE = "shared/utils/pykwalify_extensions.py"
LATEST_TRAINING_DATA_FORMAT_VERSION = "2.0"

DOMAIN_SCHEMA_FILE = "utils/schemas/domain.yml"

DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES = 0
DEFAULT_CARRY_OVER_SLOTS_TO_NEW_SESSION = True

DEFAULT_NLU_FALLBACK_INTENT_NAME = "nlu_fallback"