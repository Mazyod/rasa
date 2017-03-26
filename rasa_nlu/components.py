import inspect

from typing import Optional
from typing import Type


def load_component(component_clz, context, config):
    # type: (Type[Component], dict, dict) -> Optional[Component]
    """Calls a components load method to init it based on a previously persisted model."""

    if component_clz is not None:
        load_args = fill_args(component_clz.load_args(), context, config)
        return component_clz.load(*load_args)
    else:
        return None


def init_component(component, context, config):
    # type: (Component, dict, dict) -> None
    """Initializes a component using the attributes from the context and configuration."""

    args = fill_args(component.pipeline_init_args(), context, config)
    updates = component.pipeline_init(*args)
    if updates:
        context.update(updates)


def fill_args(arguments, context, config):
    # type: ([str], dict, dict) -> [object]
    """Given a list of arguments, tries to look up these argument names in the config / context to fill the arguments"""

    filled = []
    for arg in arguments:
        if arg in context:
            filled.append(context[arg])
        elif arg in config:
            filled.append(config[arg])
        else:
            raise MissingArgumentError("Couldn't fill argument '{}' :(".format(arg))
    return filled


class MissingArgumentError(ValueError):
    """Raised when a function is called and not all parameters can be filled from the context / config.

    Attributes:
        message -- explanation of which parameter is missing
    """

    def __init__(self, message):
        # type: (str) -> None
        super(MissingArgumentError, self).__init__(message)


class Component(object):
    """A component is a message processing unit in a pipeline.

    Components are collected sequentially in a pipeline. Each component is called one after another. This holds for
     initialization, training, persisting and loading the components. If a component comes first in a pipeline, its
     methods will be called first.

    E.g. to process an incoming message, the `process` method of each component will be called. During the processing
     (as well as the training, persisting and initialization) components can pass information to other components.
     The information is passed to other components by providing attributes to the so called pipeline context. The
     pipeline context contains all the information of the previous components a component can use to do its own
     processing. For example, a featurizer component can provide features that are used by another component down
     the pipeline to do intent classification."""

    # Name of the component to be used when integrating it in a pipeline. E.g. `[ComponentA, ComponentB]`
    # will be a proper pipeline definition where `ComponentA` is the name of the first component of the pipeline.
    name = ""

    # Defines what attributes the pipeline component will provide when called. The different keys indicate the
    # different functions (`pipeline_init`, `train`, `process`) that are able to update the pipelines context.
    # (mostly used to check if the pipeline is valid)
    context_provides = {
        "pipeline_init": [],
        "train": [],
        "process": [],
    }

    # Defines which of the attributes the component provides should be added to the final output json at the end of the
    # pipeline. Every attribute in `output_provides` should be part of the above `context_provides['process']`. As it
    # wouldn't make much sense to keep an attribute in the output that is not generated. Every other attribute provided
    # in the context during the process step will be removed from the output json.
    output_provides = []

    @classmethod
    def load(cls, *args):
        # type: (...) -> 'cls'
        """Load this component from file.

        After a component got trained, it will be persisted by calling `persist`. When the pipeline gets loaded again,
         this component needs to be able to restore itself. Components can rely on any context attributes that are
         created by `pipeline_init` calls to components previous to this one."""
        return cls()

    def pipeline_init(self, *args):
        # type: (...) -> Optional[dict]
        """Initialize this component for a new pipeline

        This function will be called before the training is started and before the first message is processed using
        the interpreter. The component gets the opportunity to add information to the context that is passed through
        the pipeline during training and message parsing. Most components do not need to implement this method.
        It's mostly used to initialize framework environments like MITIE and spacy
        (e.g. loading word vectors for the pipeline)."""
        pass

    def train(self, *args):
        # type: (...) -> Optional[dict]
        """Train this component.

        This is the components chance to train itself provided with the training data. The component can rely on
        any context attribute to be present, that gets created by a call to `pipeline_init` of ANY component and
        on any context attributes created by a call to `train` of components previous to this one."""
        pass

    def process(self, *args):
        # type: (...) -> Optional[dict]
        """Process an incomming message.

       This is the components chance to process an incommng message. The component can rely on
       any context attribute to be present, that gets created by a call to `pipeline_init` of ANY component and
       on any context attributes created by a call to `process` of components previous to this one."""
        pass

    def persist(self, model_dir):
        # type: (str) -> Optional[dict]
        """Persist this component to disk for future loading."""
        pass

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[str]
        """This key is used to cache components.

        If a component is unique to a model it should return None. Otherwise, an instantiation of the
        component will be reused for all models where the metadata creates the same key."""
        from rasa_nlu.model import Metadata

        return None

    def pipeline_init_args(self):
        # type: () -> [str]
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.pipeline_init).args)

    def train_args(self):
        # type: () -> [str]
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.train).args)

    def process_args(self):
        # type: () -> [str]
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.process).args)

    @classmethod
    def load_args(cls):
        # type: () -> [str]
        return filter(lambda arg: arg not in ["cls"], inspect.getargspec(cls.load).args)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
