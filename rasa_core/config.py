from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core import utils


def read_config(filename):
    utils.read_yaml_file(filename)
