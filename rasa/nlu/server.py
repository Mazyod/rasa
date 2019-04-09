import argparse
import asyncio
import logging
import os
from functools import wraps
from inspect import isawaitable
from sanic import Sanic, response
from sanic.request import Request
from sanic_cors import CORS
from typing import Any, Callable, Optional, Text
import simplejson

import rasa
import rasa.utils.io
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.nlu import config, utils, constants
import rasa.nlu.cli.server as cli
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.data_router import (
    DataRouter, InvalidProjectError, MaxTrainingError)
from rasa.nlu.train import TrainingException
from rasa.nlu.utils import read_endpoints, json_to_string

logger = logging.getLogger(__name__)


class ErrorResponse(Exception):
    def __init__(self, status, reason, message, details=None, help_url=None):
        self.error_info = {
            "version": rasa.__version__,
            "status": "failure",
            "message": message,
            "reason": reason,
            "details": details or {},
            "help": help_url,
            "code": status
        }
        self.status = status


def _docs(sub_url: Text) -> Text:
    """Create a url to a subpart of the docs."""
    return constants.DOCS_BASE_URL + sub_url


def create_argument_parser():
    parser = argparse.ArgumentParser(description='parse incoming text')

    cli.add_server_arguments(parser)
    utils.add_logging_option_arguments(parser)

    return parser


def requires_auth(app: Sanic,
                  token: Optional[Text] = None
                  ) -> Callable[[Any], Any]:
    """Wraps a request handler with token authentication."""

    def decorator(f: Callable[[Any, Any, Any], Any]
                  ) -> Callable[[Any, Any], Any]:
        def sender_id_from_args(args: Any,
                                kwargs: Any) -> Optional[Text]:
            argnames = utils.arguments_of(f)
            try:
                sender_id_arg_idx = argnames.index("sender_id")
                if "sender_id" in kwargs:  # try to fetch from kwargs first
                    return kwargs["sender_id"]
                if sender_id_arg_idx < len(args):
                    return args[sender_id_arg_idx]
                return None
            except ValueError:
                return None

        def sufficient_scope(request,
                             *args: Any,
                             **kwargs: Any) -> Optional[bool]:
            jwt_data = request.app.auth.extract_payload(request)
            user = jwt_data.get("user", {})

            username = user.get("username", None)
            role = user.get("role", None)

            if role == "admin":
                return True
            elif role == "user":
                sender_id = sender_id_from_args(args, kwargs)
                return sender_id is not None and username == sender_id
            else:
                return False

        @wraps(f)
        async def decorated(request: Request,
                            *args: Any,
                            **kwargs: Any) -> Any:

            provided = utils.default_arg(request, 'token', None)
            # noinspection PyProtectedMember
            if token is not None and provided == token:
                result = f(request, *args, **kwargs)
                if isawaitable(result):
                    result = await result
                return result
            elif (app.config.get('USE_JWT') and
                  request.app.auth.is_authenticated(request)):
                if sufficient_scope(request, *args, **kwargs):
                    result = f(request, *args, **kwargs)
                    if isawaitable(result):
                        result = await result
                    return result
                raise ErrorResponse(
                    403, "NotAuthorized",
                    "User has insufficient permissions.",
                    help_url=_docs(
                        "/server.html#security-considerations"))
            elif token is None and app.config.get('USE_JWT') is None:
                # authentication is disabled
                result = f(request, *args, **kwargs)
                if isawaitable(result):
                    result = await result
                return result
            raise ErrorResponse(
                401, "NotAuthenticated", "User is not authenticated.",
                help_url=_docs("/server.html#security-considerations"))

        return decorated

    return decorator


def dump_to_data_file(data):
    if isinstance(data, str):
        data_string = data
    else:
        data_string = utils.json_to_string(data)

    return utils.create_temporary_file(data_string, "_training_data")


def _configure_logging(loglevel, logfile):
    logging.basicConfig(filename=logfile,
                        level=loglevel)
    logging.captureWarnings(True)


def _load_default_config(path):
    if path:
        return config.load(path).as_dict()
    else:
        return {}


# configure async loop logging
async def configure_logging():
    if logger.isEnabledFor(logging.DEBUG):
        utils.enable_async_loop_debugging(asyncio.get_event_loop())


def create_app(data_router,
               loglevel='INFO',
               logfile=None,
               token=None,
               cors_origins=None,
               default_config_path=None):
    """Class representing Rasa NLU http server"""

    app = Sanic(__name__)
    CORS(app,
         resources={r"/*": {"origins": cors_origins or ""}},
         automatic_options=True)

    _configure_logging(loglevel, logfile)

    default_model_config = _load_default_config(default_config_path)

    @app.get("/")
    async def hello(request):
        """Main Rasa route to check if the server is online"""
        return response.text("hello from Rasa NLU: " + rasa.__version__)

    async def parse_response(request_params):
        data = data_router.extract(request_params)
        try:
            return response.json(await data_router.parse(data),
                                 status=200)
        except InvalidProjectError as e:
            return response.json({"error": "{}".format(e)},
                                 status=404)
        except Exception as e:
            logger.exception(e)
            return response.json({"error": "{}".format(e)},
                                 status=500)

    @app.get("/parse")
    @requires_auth(app, token)
    async def parse(request):
        request_params = request.raw_args

        if 'query' in request_params:
            request_params['q'] = request_params.pop('query')
        if 'q' not in request_params:
            request_params['q'] = ""
        return await parse_response(request_params)

    @app.post("/parse")
    @requires_auth(app, token)
    async def parse(request):
        request_params = request.json

        if 'query' in request_params:
            request_params['q'] = request_params.pop('query')

        if 'q' not in request_params:
            return response.json({
                "error": "Invalid parse parameter specified"},
                status=404)
        else:
            return await parse_response(request_params)

    @app.get("/version")
    @requires_auth(app, token)
    async def version(request):
        """Returns the Rasa server's version"""

        return response.json({
            "version": rasa.__version__,
            "minimum_compatible_version": MINIMUM_COMPATIBLE_VERSION
        })

    @app.get("/status")
    @requires_auth(app, token)
    async def status(request):
        return response.json(data_router.get_status())

    def extract_json(content):
        # test if json has config structure
        json_config = simplejson.loads(content).get("data")

        # if it does then this results in correct format.
        if json_config:
            return simplejson.loads(content), json_config

        # otherwise use defaults.
        else:
            return default_model_config, content

    def extract_data_and_config(request):

        request_content = request.body.decode('utf-8', 'strict')

        if 'yml' in request.content_type:
            # assumes the user submitted a model configuration with a data
            # parameter attached to it

            model_config = rasa.utils.io.read_yaml(request_content)
            data = model_config.get("data")

        elif 'json' in request.content_type:
            model_config, data = extract_json(request_content)

        else:
            raise Exception("Content-Type must be 'application/x-yml' "
                            "or 'application/json'")

        return model_config, data

    def get_request_content_type(self, request):
        content_type = request.requestHeaders.getRawHeaders("Content-Type", [])

        if len(content_type) is not 1:
            raise Exception("The request must have exactly one content type")
        else:
            return content_type[0]

    @app.post("/train")
    @requires_auth(app, token)
    async def train(request):

        # if not set will use the default project name, e.g. "default"
        project = request.raw_args.get("project", None)
        # if set will not generate a model name but use the passed one
        model_name = request.raw_args.get("model", None)

        try:
            model_config, data = extract_data_and_config(request)
        except Exception as e:
            return response.json({"error": "{}".format(e)}, status=400)

        data_file = dump_to_data_file(data)

        try:
            path_to_model = await data_router.start_train_process(
                data_file, project,
                RasaNLUModelConfig(model_config), model_name)
            zipped_path = utils.zip_folder(path_to_model)
            return await response.file(zipped_path)
        except MaxTrainingError as e:
            return response.json({"error": "{}".format(e)}, status=403)
        except InvalidProjectError as e:
            return response.json({"error": "{}".format(e)}, status=404)
        except TrainingException as e:
            return response.json({"error": "{}".format(e)}, status=500)

    @app.post("/evaluate")
    @requires_auth(app, token)
    async def evaluate(request):
        data_string = request.body.decode('utf-8', 'strict')

        try:
            payload = await data_router.evaluate(
                data_string,
                request.raw_args.get('project'),
                request.raw_args.get('model'))
            return response.json(payload)
        except Exception as e:
            return response.json({"error": "{}".format(e)}, status=500)

    @app.delete("/models")
    @requires_auth(app, token)
    async def unload_model(request):
        try:
            payload = await data_router.unload_model(
                request.raw_args.get('project',
                                     RasaNLUModelConfig.DEFAULT_PROJECT_NAME),
                request.raw_args.get('model')
            )
            return response.json(payload)
        except Exception as e:
            logger.exception(e)
            return response.json({"error": "{}".format(e)}, status=500)

    return app


def get_token(_clitoken: str) -> str:
    _envtoken = os.environ.get("RASA_NLU_TOKEN")

    if _clitoken and _envtoken:
        raise Exception(
            "RASA_NLU_TOKEN is set both with the -t option,"
            " with value `{}`, and with an environment variable, "
            "with value `{}`. "
            "Please set the token with just one method "
            "to avoid unexpected behaviours.".format(
                _clitoken, _envtoken))

    token = _clitoken or _envtoken
    return token


def main(args):
    pre_load = args.pre_load

    _endpoints = read_endpoints(args.endpoints)

    router = DataRouter(
        args.path,
        args.max_training_processes,
        args.response_log,
        args.emulate,
        args.storage,
        model_server=_endpoints.model,
        wait_time_between_pulls=args.wait_time_between_pulls)

    if pre_load:
        logger.debug('Preloading....')
        if 'all' in pre_load:
            pre_load = router.project_store.keys()
        router._pre_load(pre_load)

    rasa = create_app(
        router,
        args.loglevel,
        args.write,
        get_token(args.token),
        args.cors,
        default_config_path=args.config
    )
    rasa.add_task(configure_logging)

    logger.info('Started http server on port %s' % args.port)

    rasa.run(host='0.0.0.0', port=args.port, workers=1,
             access_log=logger.isEnabledFor(logging.DEBUG))


if __name__ == '__main__':
    raise RuntimeError("Calling `rasa.nlu.server` directly is "
                       "no longer supported. "
                       "Please use `rasa run nlu` instead.")
