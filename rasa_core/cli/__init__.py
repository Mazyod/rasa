import rasa_core.cli.arguments
import rasa_core.cli.test
import rasa_core.cli.run
import rasa_core.cli.train
import rasa_core.cli.visualization


def stories_from_cli_args(cmdline_arguments):
    from rasa_core import utils

    if cmdline_arguments.url:
        return utils.download_file_from_url(cmdline_arguments.url)
    else:
        return cmdline_arguments.stories
