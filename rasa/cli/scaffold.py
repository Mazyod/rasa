import os

import questionary
from rasa.cli import train
from rasa.cli.shell import shell
from rasa.cli.train import create_default_output_path


def add_subparser(subparsers, parents):
    scaffold_parser = subparsers.add_parser(
        "init",
        parents=parents,
        help="Create a new project from a initial_project")
    scaffold_parser.set_defaults(func=run)


def scaffold_path():
    import pkg_resources
    return pkg_resources.resource_filename(__name__, "initial_project")


def print_train_or_instructions(args, path):
    from rasa_core.utils import print_success

    print_success("Your bot is ready to go!")
    should_train = questionary.confirm("Do you want me to train an initial "
                                       "model for the bot? 💪🏽").ask()
    if should_train:
        args.config = os.path.join(path, "config.yml")
        args.stories = os.path.join(path, "data/core")
        args.domain = os.path.join(path, "domain.yml")
        args.nlu = os.path.join(path, "data/nlu")
        args.out = os.path.join(path, create_default_output_path())

        args.model = train.train(args)

        print_run_or_instructions(args, path)

    else:
        print("No problem 👍🏼. You can also train me later by going to the "
              "project directory and running 'rasa train'."
              "".format(path))


def print_run_or_instructions(args, path):
    from rasa_core import constants

    should_run = questionary.confirm("Do you want to speak to the trained bot "
                                     "on the command line? 🤖").ask()

    if should_run:
        # provide defaults for command line arguments
        attributes = ["endpoints", "credentials", "cors", "auth_token",
                      "jwt_secret", "jwt_method", "enable_api"]
        for a in attributes:
            setattr(args, a, None)

        args.port = constants.DEFAULT_SERVER_PORT

        shell(args)
    else:
        print("Ok 👍🏼. If you want to speak to the bot later, change into the"
              "project directory and run 'rasa shell'."
              "".format(path))


def init_project(args, path):
    from distutils.dir_util import copy_tree

    copy_tree(scaffold_path(), path)

    print("Created project directory at '{}'.".format(os.path.abspath(path)))
    print_train_or_instructions(args, path)


def print_cancel():
    print("Ok. Then I stop here. If you need me again, simply type "
          "'rasa init' 🙋🏽‍♀️")
    exit(0)


def run(args):
    from rasa_core.utils import print_success

    print_success("Welcome to Rasa! 🤖\n")
    print("To get started quickly, I can assist you to create an "
          "initial project.\n"
          "If you need some help to get from this template to a "
          "bad ass contextual assistant, checkout our quickstart guide"
          "here: https://rasa.com/docs/core/quickstart \n\n"
          "Now let's start! 👇🏽\n")
    path = questionary.text("Please enter a folder path where I should create "
                            "the initial project [default: current directory]",
                            default=".").ask()

    if not os.path.isdir(path):
        should_create = questionary.confirm("Path '{}' does not exist 🧐. "
                                            "Should I create it?"
                                            "".format(path)).ask()
        if should_create:
            os.makedirs(path)
        else:
            print("Ok. Then I stop here. If you need me again, simply type "
                  "'rasa init' 🙋🏽‍♀️")
            exit(0)

    if path is None or not os.path.isdir(path):
        print_cancel()

    if len(os.listdir(path)) > 0:
        overwrite = questionary.confirm(
            "Directory '{}' is not empty. Continue?"
            "".format(os.path.abspath(path))).ask()
        if not overwrite:
            print_cancel()

    init_project(args, path)
