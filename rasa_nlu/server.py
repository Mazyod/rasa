from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import logging
import os
import six
from functools import wraps

import simplejson
from builtins import str

from klein import Klein
from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks, returnValue

from rasa_nlu import utils
from rasa_nlu.data_router import DataRouter, InvalidProjectError, \
    AlreadyTrainingError
from rasa_nlu.train import TrainingException
from rasa_nlu.version import __version__
from rasa_nlu.utils import json_to_string

logger = logging.getLogger(__name__)


def create_argument_parser():
    parser = argparse.ArgumentParser(description='parse incoming text')

    parser.add_argument('-e', '--emulate',
                        choices=['wit', 'luis', 'dialogflow'],
                        help='which service to emulate (default: None i.e. use '
                             'simple built in format)')
    parser.add_argument('-P', '--port',
                        type=int,
                        help='port on which to run server')
    parser.add_argument('-t', '--token',
                        help="auth token. If set, reject requests which don't "
                             "provide this token as a query parameter")
    parser.add_argument('-w', '--write',
                        help='file where logs will be saved')
    parser.add_argument('--path',
                        help="path where project files will be saved")
    parser.add_argument('--max_training_processes',
                        help='Number of parallel trainings to run when '
                             'training data is submitted over HTTP.')
    parser.add_argument('--num_threads',
                        help='Number of parallel threads to use for '
                             'handling parse requests.')
    parser.add_argument('--response_log',
                        help='File to store responses delivered over'
                             'the parse endpoint.')
    parser.add_argument('--storage',
                        help='Set the remote location where models are stored. '
                             'E.g. on AWS. If nothing is configured, the '
                             'server will only serve the models that are '
                             'on disk in the configured `path`.')

    utils.add_logging_option_arguments(parser)

    return parser


def check_cors(f):
    """Wraps a request handler with CORS headers checking."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]
        origin = request.getHeader('Origin')

        if origin:
            if '*' in self.cors_origins:
                request.setHeader('Access-Control-Allow-Origin', '*')
            elif origin in self.cors_origins:
                request.setHeader('Access-Control-Allow-Origin', origin)
            else:
                request.setResponseCode(403)
                return 'forbidden'

        if request.method.decode('utf-8', 'strict') == 'OPTIONS':
            return ''  # if this is an options call we skip running `f`
        else:
            return f(*args, **kwargs)

    return decorated


def requires_auth(f):
    """Wraps a request handler with token authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]
        if six.PY3:
            token = request.args.get(b'token', [b''])[0].decode("utf8")
        else:
            token = str(request.args.get('token', [''])[0])
        if self.access_token is None or token == self.access_token:
            return f(*args, **kwargs)
        request.setResponseCode(401)
        return 'unauthorized'

    return decorated


def decode_parameters(request):
    """Make sure all the parameters have the same encoding.

    Ensures  py2 / py3 compatibility."""
    return {
        key.decode('utf-8', 'strict'): value[0].decode('utf-8', 'strict')
        for key, value in request.args.items()}


def parameter_or_default(request, name, default=None):
    """Return a parameters value if part of the request, or the default."""

    request_params = decode_parameters(request)
    return request_params.get(name, default)


class RasaNLU(object):
    """Class representing Rasa NLU http server"""

    app = Klein()

    def __init__(self,
                 data_router,
                 log_level='INFO',
                 log_file=None,
                 num_threads=1,
                 token=None,
                 cors_origins=None,
                 testing=False):

        self._configure_logging(log_level, log_file)

        # TODO: anything we want to print here?
        logger.debug("Configuration: ")

        self.default_model_config = {}
        self.data_router = data_router
        self._testing = testing
        self.cors_origins = cors_origins if cors_origins else ["*"]
        self.access_token = token
        reactor.suggestThreadPoolSize(num_threads * 5)

    @staticmethod
    def _configure_logging(log_level, log_file):
        logging.basicConfig(filename=log_file,
                            level=log_level)
        logging.captureWarnings(True)

    @app.route("/", methods=['GET', 'OPTIONS'])
    @check_cors
    def hello(self, request):
        """Main Rasa route to check if the server is online"""
        return "hello from Rasa NLU: " + __version__

    @app.route("/parse", methods=['GET', 'POST', 'OPTIONS'])
    @requires_auth
    @check_cors
    @inlineCallbacks
    def parse(self, request):
        request.setHeader('Content-Type', 'application/json')
        if request.method.decode('utf-8', 'strict') == 'GET':
            request_params = decode_parameters(request)
        else:
            request_params = simplejson.loads(
                    request.content.read().decode('utf-8', 'strict'))

        if 'query' in request_params:
            request_params['q'] = request_params.pop('query')

        if 'q' not in request_params:
            request.setResponseCode(404)
            dumped = json_to_string(
                    {"error": "Invalid parse parameter specified"})
            returnValue(dumped)
        else:
            data = self.data_router.extract(request_params)
            try:
                request.setResponseCode(200)
                response = yield (self.data_router.parse(data) if self._testing
                                  else threads.deferToThread(self.data_router.parse, data))
                returnValue(json_to_string(response))
            except InvalidProjectError as e:
                request.setResponseCode(404)
                returnValue(json_to_string({"error": "{}".format(e)}))
            except Exception as e:
                request.setResponseCode(500)
                logger.exception(e)
                returnValue(json_to_string({"error": "{}".format(e)}))

    @app.route("/version", methods=['GET', 'OPTIONS'])
    @requires_auth
    @check_cors
    def version(self, request):
        """Returns the Rasa server's version"""

        request.setHeader('Content-Type', 'application/json')
        return json_to_string({'version': __version__})

    @app.route("/config", methods=['GET', 'OPTIONS'])
    @requires_auth
    @check_cors
    def rasaconfig(self, request):
        """Returns the in-memory configuration of the Rasa server"""

        request.setHeader('Content-Type', 'application/json')
        # TODO: check if this works
        return json_to_string(self.default_model_config)

    @app.route("/status", methods=['GET', 'OPTIONS'])
    @requires_auth
    @check_cors
    def status(self, request):
        request.setHeader('Content-Type', 'application/json')
        return json_to_string(self.data_router.get_status())

    @app.route("/train", methods=['POST', 'OPTIONS'])
    @requires_auth
    @check_cors
    @inlineCallbacks
    def train(self, request):
        # TODO: allow to pass in whole model configuration, for now: use
        # default config
        model_config = self.default_model_config

        project = parameter_or_default(request, "project", default=None)

        data_string = request.content.read().decode('utf-8', 'strict')

        # update model config from get parameters
        for key, value in request.args.items():
            k = key.decode('utf-8', 'strict')
            v = value[0].decode('utf-8', 'strict')
            model_config[k] = v

        request.setHeader('Content-Type', 'application/json')

        try:
            request.setResponseCode(200)
            response = yield self.data_router.start_train_process(
                    data_string, project, model_config)
            returnValue(json_to_string({'info': 'new model trained: {}'
                                                ''.format(response)}))
        except AlreadyTrainingError as e:
            request.setResponseCode(403)
            returnValue(json_to_string({"error": "{}".format(e)}))
        except InvalidProjectError as e:
            request.setResponseCode(404)
            returnValue(json_to_string({"error": "{}".format(e)}))
        except TrainingException as e:
            request.setResponseCode(500)
            returnValue(json_to_string({"error": "{}".format(e)}))

    @app.route("/evaluate", methods=['POST'])
    @requires_auth
    @check_cors
    def evaluate(self, request):
        data_string = request.content.read().decode('utf-8', 'strict')
        params = {
            key.decode('utf-8', 'strict'): value[0].decode('utf-8', 'strict')
            for key, value in request.args.items()
        }

        request.setHeader('Content-Type', 'application/json')

        try:
            request.setResponseCode(200)
            response = self.data_router.evaluate(data_string,
                                                 params.get('project'),
                                                 params.get('model'))
            return simplejson.dumps(response)
        except Exception as e:
            request.setResponseCode(500)
            return simplejson.dumps({"error": "{}".format(e)})


if __name__ == '__main__':
    # Running as standalone python application
    cmdline_args = create_argument_parser().parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    router = DataRouter(cmdline_args.path,
                        cmdline_args.max_training_processes,
                        cmdline_args.response_log)
    rasa = RasaNLU(
            router,
            cmdline_args.path,
            cmdline_args.log_level,
            cmdline_args.log_file,
            cmdline_args.num_threads,
            cmdline_args.token,
    )

    logger.info('Started http server on port %s' % cmdline_args.port)
    rasa.app.run('0.0.0.0', cmdline_args.port)
