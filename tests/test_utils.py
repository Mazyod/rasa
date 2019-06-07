import pytest
from aioresponses import aioresponses
from rasa.utils.endpoints import EndpointConfig
from rasa.utils.validation import validate_pipeline_yaml, InvalidYamlFileError
from tests.utilities import latest_request, json_of_latest_request
from rasa.utils.common import sort_list_of_dicts_by_first_key
import rasa.utils.io


async def test_endpoint_config():
    with aioresponses() as mocked:
        endpoint = EndpointConfig(
            "https://example.com/",
            params={"A": "B"},
            headers={"X-Powered-By": "Rasa"},
            basic_auth={"username": "user", "password": "pass"},
            token="mytoken",
            token_name="letoken",
            type="redis",
            port=6379,
            db=0,
            password="password",
            timeout=30000,
        )

        mocked.post(
            "https://example.com/test?A=B&P=1&letoken=mytoken",
            payload={"ok": True},
            repeat=True,
            status=200,
        )

        await endpoint.request(
            "post",
            subpath="test",
            content_type="application/text",
            json={"c": "d"},
            params={"P": "1"},
        )

        r = latest_request(
            mocked, "post", "https://example.com/test?A=B&P=1&letoken=mytoken"
        )

        assert r

        assert json_of_latest_request(r) == {"c": "d"}
        assert r[-1].kwargs.get("params", {}).get("A") == "B"
        assert r[-1].kwargs.get("params", {}).get("P") == "1"
        assert r[-1].kwargs.get("params", {}).get("letoken") == "mytoken"

        # unfortunately, the mock library won't report any headers stored on
        # the session object, so we need to verify them separately
        async with endpoint.session() as s:
            assert s._default_headers.get("X-Powered-By") == "Rasa"
            assert s._default_auth.login == "user"
            assert s._default_auth.password == "pass"


def test_sort_dicts_by_keys():
    test_data = [{"Z": 1}, {"A": 10}]

    expected = [{"A": 10}, {"Z": 1}]
    actual = sort_list_of_dicts_by_first_key(test_data)

    assert actual == expected


def test_validate_pipeline_yaml():
    # should raise no exception
    validate_pipeline_yaml(
        rasa.utils.io.read_file("examples/restaurantbot/domain.yml"),
        "core/schemas/domain.yml",
    )

    validate_pipeline_yaml(
        rasa.utils.io.read_file("sample_configs/config_defaults.yml"),
        "nlu/schemas/config.yml",
    )

    validate_pipeline_yaml(
        rasa.utils.io.read_file("sample_configs/config_supervised_embeddings.yml"),
        "nlu/schemas/config.yml",
    )

    validate_pipeline_yaml(
        rasa.utils.io.read_file("sample_configs/config_crf_custom_features.yml"),
        "nlu/schemas/config.yml",
    )


def test_validate_pipeline_yaml_fails_on_invalid_domain():
    with pytest.raises(InvalidYamlFileError):
        validate_pipeline_yaml(
            rasa.utils.io.read_file("data/test_domains/invalid_format.yml"),
            "core/schemas/domain.yml",
        )


def test_validate_pipeline_yaml_fails_on_nlu_data():
    with pytest.raises(InvalidYamlFileError):
        validate_pipeline_yaml(
            rasa.utils.io.read_file("examples/restaurantbot/data/nlu.md"),
            "core/schemas/domain.yml",
        )


def test_validate_pipeline_yaml_fails_on_missing_keys():
    with pytest.raises(InvalidYamlFileError):
        validate_pipeline_yaml(
            rasa.utils.io.read_file("data/test_config/example_config.yaml"),
            "nlu/schemas/config.yml",
        )
