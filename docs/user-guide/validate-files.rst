:desc: Check the architecture to understand how Rasa Core uses machine
       learning, context and state of the conversation to predict the
       next action of the AI Assistant.

.. _validate_files:

Validate Files
==============


Test files for mistakes
-----------------------

To verify if there are any mistakes in your domain file, nlu or story data, run the validate script.
You can run it with the following command:

.. code-block:: bash

  $ rasa data validate -s data/stories.md -d domain.yml -u data/nlu.md

The script above runs all the validations on your files. Here is the list of options to
the script:

.. program-output:: rasa data validate --help 

You can also run these validations through the Python API by importing the `Validator` class,
which has the following methods:

**from_files():** Creates the instance from string paths to the necessary files.

**verify_intents():** Checks if intents listed in domain file are consistent with the nlu data.

**verify_intents_in_stories():** Verification for intents in the stories, to check if they are valid.

**verify_utterances():** Checks for domain file for consistency between utterance templates and those listed in the
actions.

**verify_utterances_in_stories():** Verification for utterances in stories, to check if they are valid.

**verify_all():** Runs all verifications above.

To use these functions it is necessary to create a `Validator` object and initialize the logger. See the following code:

.. code-block:: python

  import logging
  from rasa import utils
  from rasa.core.validate import Validate

  logger = logging.getLogger(__name__)

  utils.configure_colored_logging('DEBUG')

  validator = Validator.from_files(domain_file='domain.yml',
                                   nlu_data='data/nlu_data.md',
                                   stories='data/stories.md')

  validator.verify_all().