import asyncio
import multiprocessing
from asyncio import Task

import datetime
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Text

from rasa_nlu import config, utils
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.emulators import NoEmulator
from rasa_nlu.evaluate import run_evaluation
from rasa_nlu.model import InvalidProjectError
from rasa_nlu.project import (
    Project, STATUS_FAILED, STATUS_READY, STATUS_TRAINING, load_from_server)
from rasa_nlu.train import do_train_in_worker

logger = logging.getLogger(__name__)

# in some execution environments `reactor.callFromThread`
# can not be called as it will result in a deadlock as
# the `callFromThread` queues the function to be called
# by the reactor which only happens after the call to `yield`.
# Unfortunately, the test is blocked there because `app.flush()`
# needs to be called to allow the fake server to
# respond and change the status of the Deferred on which
# the client is yielding. Solution: during tests we will set
# this Flag to `False` to directly run the calls instead
# of wrapping them in `callFromThread`.
DEFERRED_RUN_IN_REACTOR_THREAD = True


class MaxTrainingError(Exception):
    """Raised when a training is requested and the server has
        reached the max count of training processes.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = 'The server can\'t train more models right now!'

    def __str__(self):
        return self.message


class DataRouter(object):
    def __init__(self,
                 project_dir=None,
                 max_training_processes=1,
                 response_log=None,
                 emulation_mode=None,
                 remote_storage=None,
                 component_builder=None,
                 model_server=None,
                 wait_time_between_pulls=None):
        self._training_processes = max(max_training_processes, 1)
        self._current_training_processes = 0
        self.responses = self._create_query_logger(response_log)
        self.project_dir = config.make_path_absolute(project_dir)
        self.emulator = self._create_emulator(emulation_mode)
        self.remote_storage = remote_storage
        self.model_server = model_server
        self.wait_time_between_pulls = wait_time_between_pulls

        if component_builder:
            self.component_builder = component_builder
        else:
            self.component_builder = ComponentBuilder(use_cache=True)

        self.project_store = self._create_project_store(project_dir)

        # tensorflow sessions are not fork-safe,
        # and training processes have to be spawned instead of forked. See
        # https://github.com/tensorflow/tensorflow/issues/5448#issuecomment
        # -258934405
        multiprocessing.set_start_method('spawn', force=True)

        self.pool = ProcessPoolExecutor(
            max_workers=self._training_processes)

    def __del__(self):
        """Terminates workers pool processes"""
        self.pool.shutdown()

    @staticmethod
    def _create_query_logger(response_log):
        """Create a logger that will persist incoming query results."""

        # Ensures different log files for different
        # processes in multi worker mode
        if response_log:
            # We need to generate a unique file name,
            # even in multiprocess environments
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file_name = "rasa_nlu_log-{}-{}.log".format(timestamp,
                                                            os.getpid())
            response_logfile = os.path.join(response_log, log_file_name)
            # Instantiate a standard python logger,
            # which we are going to use to log requests
            utils.create_dir_for_file(response_logfile)
            # noinspection PyTypeChecker
            query_logger = logging.getLogger('query-logger')
            query_logger.setLevel(logging.INFO)
            ch = logging.FileHandler(response_logfile)
            ch.setFormatter(logging.Formatter('%(message)s'))
            query_logger.propagate = False
            query_logger.addHandler(ch)
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info("Logging of requests is disabled. "
                        "(No 'request_log' directory configured)")
            return None

    def _collect_projects(self, project_dir: Text) -> List[Text]:
        if project_dir and os.path.isdir(project_dir):
            logger.debug("Listing projects in '{}'".format(project_dir))
            projects = os.listdir(project_dir)
        else:
            projects = []

        projects.extend(self._list_projects_in_cloud())
        return projects

    def _create_project_store(self,
                              project_dir: Text) -> Dict[Text, Any]:
        default_project = RasaNLUModelConfig.DEFAULT_PROJECT_NAME

        projects = self._collect_projects(project_dir)

        project_store = {}

        if self.model_server is not None:
            project_store[default_project] = load_from_server(
                self.component_builder,
                default_project,
                self.project_dir,
                self.remote_storage,
                self.model_server,
                self.wait_time_between_pulls
            )
        else:
            for project in projects:
                project_store[project] = Project(self.component_builder,
                                                 project,
                                                 self.project_dir,
                                                 self.remote_storage)

            if not project_store:
                project_store[default_project] = Project(
                    project=default_project,
                    project_dir=self.project_dir,
                    remote_storage=self.remote_storage
                )

        return project_store

    def _pre_load(self, projects: List[Text]) -> None:
        logger.debug("loading %s", projects)
        for project in self.project_store:
            if project in projects:
                self.project_store[project].load_model()

    def _list_projects_in_cloud(self) -> List[Text]:
        # noinspection PyBroadException
        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(self.remote_storage)
            if p is not None:
                return p.list_projects()
            else:
                return []
        except Exception:
            logger.exception("Failed to list projects. Make sure you have "
                             "correctly configured your cloud storage "
                             "settings.")
            return []

    @staticmethod
    def _create_emulator(mode: Optional[Text]) -> NoEmulator:
        """Create emulator for specified mode.

        If no emulator is specified, we will use the Rasa NLU format."""

        if mode is None:
            return NoEmulator()
        elif mode.lower() == 'wit':
            from rasa_nlu.emulators.wit import WitEmulator
            return WitEmulator()
        elif mode.lower() == 'luis':
            from rasa_nlu.emulators.luis import LUISEmulator
            return LUISEmulator()
        elif mode.lower() == 'dialogflow':
            from rasa_nlu.emulators.dialogflow import DialogflowEmulator
            return DialogflowEmulator()
        else:
            raise ValueError("unknown mode : {0}".format(mode))

    @staticmethod
    def _tf_in_pipeline(model_config: RasaNLUModelConfig) -> bool:
        from rasa_nlu.classifiers.embedding_intent_classifier import \
            EmbeddingIntentClassifier
        return EmbeddingIntentClassifier.name in model_config.component_names

    def extract(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        return self.emulator.normalise_request_json(data)

    async def parse(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        project = data.get("project", RasaNLUModelConfig.DEFAULT_PROJECT_NAME)
        model = data.get("model")

        if project not in self.project_store:
            projects = self._list_projects(self.project_dir)

            cloud_provided_projects = self._list_projects_in_cloud()
            projects.extend(cloud_provided_projects)

            if project not in projects:
                raise InvalidProjectError(
                    "No project found with name '{}'.".format(project))
            else:
                try:
                    self.project_store[project] = Project(
                        self.component_builder, project,
                        self.project_dir, self.remote_storage)
                except Exception as e:
                    raise InvalidProjectError(
                        "Unable to load project '{}'. "
                        "Error: {}".format(project, e))

        time = data.get('time')
        response = self.project_store[project].parse(data['text'], time,
                                                     model)

        if self.responses:
            self.responses.info('', user_input=response, project=project,
                                model=response.get('model'))

        return self.format_response(response)

    @staticmethod
    def _list_projects(path: Text) -> List[Text]:
        """List the projects in the path, ignoring hidden directories."""
        return [os.path.basename(fn)
                for fn in utils.list_subdirectories(path)]

    def format_response(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        return self.emulator.normalise_response_json(data)

    def get_status(self) -> Dict[Text, Any]:
        # This will only count the trainings started from this
        # process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.

        return {
            "max_training_processes": self._training_processes,
            "current_training_processes": self._current_training_processes,
            "available_projects": {
                name: project.as_dict()
                for name, project in self.project_store.items()
            }
        }

    async def start_train_process(self,
                                  data_file: Text,
                                  project: Text,
                                  train_config: RasaNLUModelConfig,
                                  model_name: Optional[Text] = None
                                  ):
        """Start a model training."""

        if not project:
            raise InvalidProjectError("Missing project name to train")

        if self._training_processes <= self._current_training_processes:
            raise MaxTrainingError

        if project in self.project_store:
            self.project_store[project].status = STATUS_TRAINING
        elif project not in self.project_store:
            self.project_store[project] = Project(
                self.component_builder, project,
                self.project_dir, self.remote_storage)
            self.project_store[project].status = STATUS_TRAINING

        loop = asyncio.get_event_loop()

        logger.debug("New training queued")

        self._current_training_processes += 1
        self.project_store[project].current_training_processes += 1

        task: Task = loop.run_in_executor(self.pool,
                                          do_train_in_worker,
                                          train_config,
                                          data_file,
                                          self.project_dir,
                                          project,
                                          model_name,
                                          self.remote_storage)

        try:
            model_path = await task
            model_dir = os.path.basename(os.path.normpath(model_path))
            self.project_store[project].update(model_dir)

            if (self.project_store[project].status == STATUS_TRAINING and
                    not self.project_store[project].current_training_processes):
                self.project_store[project].status = STATUS_READY
            return model_dir
        except Exception as e:
            logger.warning(e)
            self.project_store[project].status = STATUS_FAILED
            self.project_store[project].error_message = str(e)

            raise
        finally:
            self._current_training_processes -= 1
            self.project_store[project].current_training_processes -= 1

    # noinspection PyProtectedMember
    async def evaluate(self,
                       data: Text,
                       project: Optional[Text] = None,
                       model: Optional[Text] = None) -> Dict[Text, Any]:
        """Perform a model evaluation."""

        project = project or RasaNLUModelConfig.DEFAULT_PROJECT_NAME
        model = model or None
        file_name = utils.create_temporary_file(data, "_training_data")

        if project not in self.project_store:
            raise InvalidProjectError("Project {} could not "
                                      "be found".format(project))

        model_name = self.project_store[project]._dynamic_load_model(model)

        self.project_store[project]._loader_lock.acquire()
        try:
            if not self.project_store[project]._models.get(model_name):
                interpreter = self.project_store[project]. \
                    _interpreter_for_model(model_name)
                self.project_store[project]._models[model_name] = interpreter
        finally:
            self.project_store[project]._loader_lock.release()

        return run_evaluation(
            data_path=file_name,
            model=self.project_store[project]._models[model_name],
            errors_filename=None
        )

    async def unload_model(self,
                           project: Optional[Text],
                           model: Text) -> Dict[Text, Any]:
        """Unload a model from server memory."""

        if project is None:
            raise InvalidProjectError("No project specified")
        elif project not in self.project_store:
            raise InvalidProjectError("Project {} could not "
                                      "be found".format(project))

        try:
            unloaded_model = self.project_store[project].unload(model)
            return unloaded_model
        except KeyError:
            raise InvalidProjectError("Failed to unload model {} "
                                      "for project {}.".format(model, project))
