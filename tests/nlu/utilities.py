import tempfile
import ruamel.yaml as yaml

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter
from rasa.nlu.train import train


def base_test_conf(pipeline_template):
    return RasaNLUModelConfig({"pipeline": pipeline_template})


def write_file_config(file_config):
    with tempfile.NamedTemporaryFile(
        "w+", suffix="_tmp_config_file.yml", delete=False
    ) as f:
        f.write(yaml.safe_dump(file_config))
        f.flush()
        return f


async def interpreter_for(component_builder, data, path, config):
    (trained, _, path) = await train(
        config, data, path, component_builder=component_builder
    )
    interpreter = Interpreter.load(path, component_builder)
    return interpreter


class ResponseTest:
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload
