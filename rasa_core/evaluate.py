from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import str

import argparse
import io
import logging
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import os
import json
import numpy as np
import pickle
from collections import defaultdict

from rasa_core import training
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.events import ActionExecuted
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.generator import TrainingDataGenerator
from rasa_core.utils import AvailableEndpoints
from rasa_nlu.evaluate import (
    plot_confusion_matrix,
    get_evaluation_metrics)
from rasa_nlu.utils import list_subdirectories

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser for the evaluate script."""

    parser = argparse.ArgumentParser(
            description='evaluates a dialogue model')
    parser.add_argument(
            '-s', '--stories',
            type=str,
            required=True,
            help="file or folder containing stories to evaluate on")
    parser.add_argument(
            '-m', '--max_stories',
            type=int,
            help="maximum number of stories to test on")
    parser.add_argument(
            '-d', '--core',
            type=str,
            help="core model to run with the server")
    parser.add_argument(
            '-u', '--nlu',
            type=str,
            help="nlu model to run with the server. None for regex interpreter")
    parser.add_argument(
            '-o', '--output',
            type=str,
            default="story_confmat.pdf",
            help="output path for the created evaluation plot. If set to None"
                 "or an empty string, no plot will be generated.")
    parser.add_argument(
            '--failed',
            type=str,
            default="failed_stories.md",
            help="output path for the failed stories")
    parser.add_argument(
            '--endpoints',
            default=None,
            help="Configuration file for the connectors as a yml file")
    parser.add_argument(
            '--fail_on_prediction_errors',
            action='store_true',
            help="If a prediction error is encountered, an exception "
                 "is thrown. This can be used to validate stories during "
                 "tests, e.g. on travis.")
    parser.add_argument(
            '--mode',
            type=str,
            default='default',
            help="default|compare (evaluate a model, or evaluate multiple "
                 "models and compare them)")
    parser.add_argument(
            '--models',
            type=str,
            help="folder containing trained models for comparison")

    utils.add_logging_option_arguments(parser)
    return parser


class WronglyPredictedAction(ActionExecuted):
    """The model predicted the wrong action.

    Mostly used to mark wrong predictions and be able to
    dump them as stories."""

    type_name = "wrong_action"

    def __init__(self, correct_action, predicted_action, timestamp=None):
        self.correct_action = correct_action
        self.predicted_action = predicted_action
        super(WronglyPredictedAction, self).__init__(correct_action,
                                                     timestamp=timestamp)

    def as_story_string(self):
        return "{}   <!-- predicted: {} -->".format(self.correct_action,
                                                    self.predicted_action)


def _generate_trackers(resource_name, agent, max_stories=None):
    story_graph = training.extract_story_graph(resource_name, agent.domain,
                                               agent.interpreter)
    g = TrainingDataGenerator(story_graph, agent.domain,
                              use_story_concatenation=False,
                              augmentation_factor=0,
                              tracker_limit=max_stories)
    return g.generate()


def _predict_tracker_actions(tracker, agent, fail_on_prediction_errors=False):
    processor = agent.create_processor()

    golds = []
    predictions = []

    events = list(tracker.events)

    partial_tracker = DialogueStateTracker.from_events(tracker.sender_id,
                                                       events[:1],
                                                       agent.domain.slots)

    for event in events[1:]:
        if isinstance(event, ActionExecuted):
            action, _, _ = processor.predict_next_action(partial_tracker)

            predicted = action.name()
            gold = event.action_name

            predictions.append(predicted)
            golds.append(gold)

            if predicted != gold:
                partial_tracker.update(WronglyPredictedAction(gold, predicted))
                if fail_on_prediction_errors:
                    raise ValueError(
                            "Model predicted a wrong action. Failed Story: "
                            "\n\n{}".format(partial_tracker.export_stories()))
            else:
                partial_tracker.update(event)
        else:
            partial_tracker.update(event)

    return golds, predictions, partial_tracker


def collect_story_predictions(completed_trackers,
                              agent,
                              fail_on_prediction_errors=False):
    """Test the stories from a file, running them through the stored model."""

    predictions = []
    golds = []
    failed = []
    correct_dialogues = []
    num_stories = len(completed_trackers)

    logger.info("Evaluating {} stories\n"
                "Progress:".format(num_stories))

    for tracker in tqdm(completed_trackers):
        current_golds, current_predictions, predicted_tracker = \
            _predict_tracker_actions(tracker, agent, fail_on_prediction_errors)

        predictions.extend(current_predictions)
        golds.extend(current_golds)

        if current_golds != current_predictions:
            # there is at least one prediction that is wrong
            failed.append(predicted_tracker)
            correct_dialogues.append(0)
        else:
            correct_dialogues.append(1)

    logger.info("Finished collecting predictions.")
    log_evaluation_table([1] * len(completed_trackers),
                         correct_dialogues,
                         "CONVERSATION",
                         include_report=False)

    return golds, predictions, failed, num_stories


def log_failed_stories(failed, failed_output):
    """Takes stories as a list of dicts"""

    if not failed_output:
        return

    with io.open(failed_output, 'w', encoding="utf-8") as f:
        if len(failed) == 0:
            f.write("<!-- All stories passed -->")
        else:
            for failure in failed:
                f.write(failure.export_stories())
                f.write("\n\n")


def run_story_evaluation(resource_name, agent,
                         max_stories=None,
                         out_file_stories=None,
                         out_file_plot=None,
                         fail_on_prediction_errors=False):
    """Run the evaluation of the stories, optionally plots the results."""

    completed_trackers = _generate_trackers(resource_name, agent, max_stories)

    test_y, predictions, failed, _ = collect_story_predictions(
            completed_trackers, agent, fail_on_prediction_errors)

    if out_file_plot:
        plot_story_evaluation(test_y, predictions, out_file_plot)

    log_failed_stories(failed, out_file_stories)


def log_evaluation_table(golds, predictions, name,
                         include_report=True):  # pragma: no cover
    """Log the sklearn evaluation metrics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        report, precision, f1, accuracy = get_evaluation_metrics(golds,
                                                                 predictions)

    logger.info("Evaluation Results on {} level:".format(name))
    logger.info("\tCorrect:   {} / {}".format(int(len(golds) * accuracy),
                                              len(golds)))
    logger.info("\tF1-Score:  {:.3f}".format(f1))
    logger.info("\tPrecision: {:.3f}".format(precision))
    logger.info("\tAccuracy:  {:.3f}".format(accuracy))

    if include_report:
        logger.info("\tClassification report: \n{}".format(report))


def plot_story_evaluation(test_y, predictions, out_file):
    """Plot the results. of story evaluation"""
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt

    log_evaluation_table(test_y, predictions, "ACTION", include_report=True)
    cnf_matrix = confusion_matrix(test_y, predictions)
    plot_confusion_matrix(cnf_matrix,
                          classes=unique_labels(test_y, predictions),
                          title='Action Confusion matrix')

    fig = plt.gcf()
    fig.set_size_inches(int(20), int(20))
    fig.savefig(out_file, bbox_inches='tight')


def run_comparison_evaluation(models, stories, output):

    """Evaluates multiple trained models on a test set"""

    num_correct = defaultdict(list)

    for run in list_subdirectories(models):
        correct_embed = []
        correct_keras = []

        for model in sorted(list_subdirectories(run)):
            logger.info("Evaluating model {}".format(model))

            agent = Agent.load(model)

            completed_trackers = _generate_trackers(stories, agent)

            actual, preds, failed_stories, no_of_stories = \
                collect_story_predictions(completed_trackers,
                                          agent)
            if 'keras' in model:
                correct_keras.append(no_of_stories - len(failed_stories))
            elif 'embed' in model:
                correct_embed.append(no_of_stories - len(failed_stories))

        num_correct['keras'].append(correct_keras)
        num_correct['embed'].append(correct_embed)

    utils.create_dir_for_file(os.path.join(output, 'results.json'))

    with io.open(os.path.join(output, 'results.json'), 'w') as f:
        json.dump(num_correct, f)


def plot_curve(output, no_stories, ax=None, **kwargs):
    """ plots the results from run_comparison_evaluation"""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()

    # load results from file
    with io.open(os.path.join(output, 'results.json')) as f:
        data = json.load(f)
    x = no_stories

    # compute mean of all the runs for keras/embed policies
    for label in ['keras', 'embed']:
        if len(data[label]) == 0:
            continue
        mean = np.mean(data[label], axis=0)
        std = np.std(data[label], axis=0)
        ax.plot(x, mean, label=label, marker='.')
        ax.fill_between(x,
                        [m-s for m, s in zip(mean, std)],
                        [m+s for m, s in zip(mean, std)],
                        color='#6b2def',
                        alpha=0.2)
    ax.legend(loc=4)
    ax.set_xlabel("Number of stories present during training")
    ax.set_ylabel("Number of correct test stories")
    plt.savefig(os.path.join(output, 'graph.pdf'), format='pdf')
    plt.show()


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.basicConfig(level=cmdline_args.loglevel)
    _endpoints = AvailableEndpoints.read_endpoints(cmdline_args.endpoints)

    if cmdline_args.mode == 'default':
        _interpreter = NaturalLanguageInterpreter.create(cmdline_args.nlu,
                                                         _endpoints.nlu)

        _agent = Agent.load(cmdline_args.core,
                            interpreter=_interpreter)
        run_story_evaluation(cmdline_args.stories,
                             _agent,
                             cmdline_args.max_stories,
                             cmdline_args.failed,
                             cmdline_args.output,
                             cmdline_args.fail_on_prediction_errors)

    elif cmdline_args.mode == 'compare':
        run_comparison_evaluation(cmdline_args.models, cmdline_args.stories,
                                  cmdline_args.output)

        no_stories = pickle.load(io.open(os.path.join(cmdline_args.models,
                                                      'num_stories.p'), 'rb'))

        plot_curve(cmdline_args.output, no_stories)
    else:
        raise ValueError("--mode can take the values default or compare")

    logger.info("Finished evaluation")
