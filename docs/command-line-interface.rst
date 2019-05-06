.. _cli-usage:

Command Line Interface
======================


.. contents:: 
   :local:

Cheat Sheet
~~~~~~~~~~~

The command line interface (CLI) gives you easy-to-remember commands for common tasks.

=========================  ===================================================================================
Command                    Effect
=========================  ===================================================================================
``rasa init``              Creates a new project, with example training data, actions, and config files.
``rasa run``               Starts a server with your model loaded. See the :ref:`http-api` docs for details.
``rasa run actions``       Starts an action server using the Rasa SDK.
``rasa shell``             Loads your trained model and lets you talk to your assistant on the command line.
``rasa train``             Trains a model using your nlu data and stories, saves trained model in ``./models``.
``rasa interactive``       Starts an interactive learning session, to create new training data by chatting.
``rasa test``              Tests a trained model using your test nlu data and stories.
``rasa -h``                Shows all available commands.
=========================  ===================================================================================

.. note::

    You can also see the available options for each subcommand 
    by adding the ``-h`` flag, e.g. ``rasa train -h``


Create a new project
~~~~~~~~~~~~~~~~~~~~

A single command sets up a complete project for you with some example training data.

.. code:: bash

   rasa init


This creates the following files:

.. code:: bash

   .
   ├── __init__.py
   ├── actions.py
   ├── config.yml
   ├── credentials.yml
   ├── data
   │   ├── nlu.md
   │   └── stories.md
   ├── domain.yml
   ├── endpoints.yml
   └── models
       └── <timestamp>.tar.gz

The ``rasa init`` command will ask you if you want to train an initial model using this data.
If you answer no, the ``models`` directory will be empty.

With this project setup, common commands are very easy to remember.
To train a model, type ``rasa train``, to talk to your model on the command line, ``rasa shell``,
to test your model type ``rasa test``. 


Start a Server
~~~~~~~~~~~~~~

To start a server running your Rasa model, run:

.. code:: bash

   rasa run

See the Rasa :ref:`http-api` docs for detailed documentation of all the endpoints.
For more information on the additional parameters, see :ref:`_section_http``


.. _run-action-server:

Start an Action Server
~~~~~~~~~~~~~~~~~~~~~~

To run your action server run

.. code:: bash

   rasa run actions


Talk to your Assistant
~~~~~~~~~~~~~~~~~~~~~~

To start a chat session with your assistant on the command line, run:

.. code:: bash

   rasa shell


In case you just have a trained NLU model in the `models` directory, `rasa shell` allows
you to obtain the intent and entities of any text you type on the command line.
If your model includes a trained Core model, you can chat with your bot and see
what he predicts as a next action.

To increase the logging level for debugging, run:

.. code:: bash

   rasa shell --debug


Train a Model
~~~~~~~~~~~~~

The main command is:

.. code:: bash

   rasa train


If you only want to train an NLU or a Core model,
you can run ``rasa train nlu`` or ``rasa train core``.
However, Rasa will automatically skip training core or nlu 
if the training data and config haven't changed.


Interactive Learning
~~~~~~~~~~~~~~~~~~~~

To start an interactive learning session with your assistant, run

.. code:: bash

   rasa interactive


This command will initially train a Rasa model with the data located in `data`. After training the
first initial model, the interactive learning session starts. However, training will be skipped if
the training data and config haven't changed.


Create a Train-Test Split
~~~~~~~~~~~~~~~~~~~~~~~~~

To create a split of your NLU data, run:

.. code:: bash

   rasa data split nlu


This will attempt to keep the proportions of intents the same in train and test.


Convert Data Between Markdown and JSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert nlu data from markdown to json (or back again), run:

.. code:: bash

   rasa data convert nlu -d data/nlu.md -o nlu.json -f json


The flags are ``rasa data convert nlu -d <INPUT_FILE> -o <OUTPUT_FILE> -f <OUTPUT_FORMAT>``.


Visualize your Stories
~~~~~~~~~~~~~~~~~~~~~~

To open a browser tab with a graph showing your stories:

.. code:: bash

   rasa show stories


.. _section_evaluation:

Evaluate a Model on Test Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate your model on test data, run:

.. code-block:: 

   rasa test

Check out more details in :ref:`nlu-evaluation` and :ref:`core-evaluation` .
