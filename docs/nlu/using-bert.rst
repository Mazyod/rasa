:desc: Find out how to use only Bert inside of Rasa NLU.

Using Bert
==========

Since Rasa 1.8.1 you can use Bert inside of Rasa pipelines.
The goal of this document is to show you how you can do that
as well as some tips in exploring these new tools.

.. contents::
   :local:

.. _using_bert:


.. edit-link::

Setup
-----

To demonstrate how to use Bert we will train two pipelines on Sara, 
the demo bot at Rasa. In doing this we will also be able to measure
the pros and cons of having Bert in your pipeline.

If you want to reproduce the results in this document you will need 
to first clone the repository found here:

.. code-block:: bash

    git clone git@github.com:RasaHQ/rasa-demo.git

Once cloned you can install the requirements. Be sure that 
you explicitly install the transformers dependency. 

.. code-block:: bash

    pip install "rasa[transformers]"

You should now be all set to train an assistant that will
use Bert. So let's write configuration files that will allow
us to compare approaches. We'll make a seperate folder 
where we can place two new configuration files. 

.. code-block:: bash

    mkdir config

For the next step we've created two configuration files. They only
contain the pipeline part that is relevant for `nlu` so no policies.

config/config-light.yml
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    language: en
    pipeline:
    - name: WhitespaceTokenizer
    - name: CountVectorsFeaturizer
    - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
    - name: DIETClassifier
    epochs: 20

config/config-heavy.yml 
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    language: en
    pipeline:
    - name: HFTransformersNLP
    model_weights: "bert-base-uncased"
    model_name: "bert"
    - name: LanguageModelTokenizer
    - name: LanguageModelFeaturizer
    - name: DIETClassifier
    epochs: 30

Note the differences in these files. 

In the light configuration we have a CountVectorsFeaturizer, which we 
replace in the heavy variant with a HFTransformersNLP together with the
LanguageModelTokenizer and LanguageModelFeaturizer. Notice that we're 
no longer using the original WhitespaceTokenizer because tokenization
is now handled by Bert.

Run the Pipelines
-----------------

You can run both configuarions yourself. 

.. code-block:: yaml

    mkdir gridresults
    rasa test nlu --config configs/config-light.yml \
                  --cross-validation --runs 1 --folds 2 \
                  --out gridresults/config-light
    rasa test nlu --config configs/config-heavy.yml \
                  --cross-validation --runs 1 --folds 2 \
                  --out gridresults/config-heavy

When this runs you should see logs appear. We've picked a few
of those lines to list them here. 

.. code-block:: txt

    # output from the light model
    2020-03-30 16:21:54 INFO     rasa.nlu.model  - Starting to train component DIETClassifier
    Epochs: 100%|███████████████████████████████| 50/50 [04:30<00:00, ...]
    2020-03-30 16:23:53 INFO     rasa.nlu.test  - Running model for predictions:
    100%|███████████████████████████████████████| 2396/2396 [01:23<00:00, 28.65it/s]
    ...
    # output from the heavy model
    2020-03-30 16:47:04 INFO     rasa.nlu.model  - Starting to train component DIETClassifier
    Epochs: 100%|███████████████████████████████| 50/50 [04:33<00:00,  ...]
    2020-03-30 16:49:52 INFO     rasa.nlu.test  - Running model for predictions:
    100%|███████████████████████████████████████| 2396/2396 [07:20<00:00,  5.69it/s]

From the logs we can gather an important observation. 
The heavy model is a fair bit slower, not in training, but at inference time
we see a ~6 fold increase. Depending on your use-case this is 
something to seriously consider.

Results
-------

We've summerised the results into two charts, one for intents and 
one for entities.


Intent Results 
~~~~~~~~~~~~~~

.. image:: /_static/images/bert-intents.png

Entity Results 
~~~~~~~~~~~~~~

.. image:: /_static/images/bert-entities.png

Observations 
~~~~~~~~~~~~

On all fronts we see that the model with the Bert embeddings performs better. 
But it deserves mentioning that the effect is more pronounced in the entities.
Note that these results may not be the same on your use-case. Every assistant 
is different so it is important that you keep comparing. 

It also deserves 
mentioning that you need to beware that you don't over-optimise training data
that you've generated yourself. End users will use the assistant in ways you 
probably did not anticipate. Typically it is more important to gather data of 
actual users than it is to get the best F1 score on an artificial dataset.