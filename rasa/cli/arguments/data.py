import argparse

from rasa.cli.arguments.default_arguments import add_nlu_data_param, add_out_param


def set_convert_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data", required=True, help="File or directory containing NLU data."
    )

    add_out_param(
        parser,
        required=True,
        default=None,
        help="File where to save training data in Rasa format.",
    )

    parser.add_argument("-l", "--language", default="en", help="Language of data.")

    parser.add_argument(
        "-f",
        "--format",
        required=True,
        choices=["json", "md"],
        help="Output format the training data should be converted into.",
    )


def set_split_arguments(parser: argparse.ArgumentParser):
    add_nlu_data_param(parser)

    parser.add_argument(
        "--training-fraction",
        type=float,
        default=0.8,
        help="Percentage of the data which should be in the training data.",
    )

    add_out_param(
        parser,
        default="train_test_split",
        help="Directory where the split files should be stored.",
    )
