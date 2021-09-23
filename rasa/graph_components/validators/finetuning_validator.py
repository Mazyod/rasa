from __future__ import annotations
from typing import Dict, Text, Any, Optional
import copy
import logging

from packaging import version
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter
import rasa.shared.utils.io

FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY = "config-without-epochs"
FINGERPRINT_CORE = "domain-without-responses"
FINGERPRINT_NLU = "nlu-labels"
FINGERPRINT_VERSION = "rasa-version"

logger = logging.getLogger(__name__)


class FineTuningValidator(GraphComponent):
    """Component that checks whether fine-tuning is possible.

    This is a component at the beginning of the graph which receives all training data
    and raises an exception in case `is_finetuning` is `True` and finetuning is not
    possible (e.g. because new labels were added).
    In case we are doing a regular training (and not finetuning) this persists the
    necessary information extracted from the training data to be able to validate when
    initialized via load whether we can finetune.

    Finetuning is possible if, compared to the initial training phase, it holds that
    1. the configuration (except for "epoch" keys) does not change
    2. the domain (except for "responses") does not change - or we're not finetuning
       the core part
    3. the intents, entities, entity groups, entity roles, and action names that
       appeared in the original NLU training data, appear in the NLU training data
       used for finetuning, and no new such items (i.e. intents, entities, entity
       groups, entity roles, or action names) have been added, compared to the original
       training data - or we're not finetuning the nlu part.
    Note that even though conditions 2. and 3. differ based on which part we finetune,
    condition 1. always covers both parts, i.e. NLU and Core.
    """

    FILENAME = "fingerprints-for-validation.json"

    def __init__(
        self,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        fingerprints: Optional[Dict[Text, Text]] = None,
    ) -> None:
        """Instantiates a `FineTuningValidator`.

        Args:
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            fingerprints: a dictionary of fingerprints generated by a
                `FineTuningValidator`
        """
        self._is_finetuning = execution_context.is_finetuning
        self._model_storage = model_storage
        self._resource = resource
        self._fingerprints: Dict[Text, Text] = fingerprints or {}

    def validate_nlu_only(self, importer: TrainingDataImporter,) -> None:
        """Validates whether we can finetune the NLU part when finetuning is enabled.

        Args:
            importer: a training data importer

        Raises:
            `InvalidConfigException` if there is a conflict
        """
        self._validate(importer=importer, nlu=True, core=False)

    def validate_core_only(self, importer: TrainingDataImporter,) -> None:
        """Validates whether we can finetune the Core part when finetuning is enabled.

        Args:
            importer: a training data importer

        Raises:
            `InvalidConfigException` if there is a conflict
        """
        self._validate(importer=importer, nlu=False, core=True)

    def validate(self, importer: TrainingDataImporter,) -> None:
        """Validates whether we can finetune Core and NLU when finetuning is enabled.

        Args:
            importer: a training data importer
        Raises:
            `InvalidConfigException` if there is a conflict
        """
        self._validate(importer, nlu=True, core=True)

    def _validate(
        self, importer: TrainingDataImporter, nlu: bool = True, core: bool = True,
    ) -> None:
        """Validate whether the finetuning setting conflicts with other settings.

        Note that this validation always takes into account the configuration of
        nlu *and* core part, while the validation of aspects of the domain and
        the NLU training data only happen if we request to validate finetuning
        with respect to NLU/Core models, respectively.

        For more details, see docstring of this class.

        Args:
            importer: a training data importer
            domain: the domain
            nlu: set to `False` if NLU part should not be validated
            core: set to `False` if Core part should not be validated
        Raises:
            `InvalidConfigException` if there is a conflict
        """
        if self._is_finetuning and not self._fingerprints:
            raise InvalidConfigException(
                f"Finetuning is enabled but the {self.__class__.__name__} "
                f"does not remember seeing a training run. Ensure that you have "
                f"trained your model at least once (with finetuning disabled) "
                f"and ensure that the  {self.__class__.__name__} is part of the "
                f"training graph. "
            )

        rasa_version = rasa.__version__
        if self._is_finetuning:
            old_rasa_version = self._fingerprints[FINGERPRINT_VERSION]
            if version.parse(old_rasa_version) < version.parse(
                MINIMUM_COMPATIBLE_VERSION
            ):
                raise InvalidConfigException(
                    f"The minimum compatible Rasa Version is "
                    f"{MINIMUM_COMPATIBLE_VERSION} but the model we attempt to "
                    f"finetune has been generated with an older version "
                    f"({old_rasa_version}."
                )
        self._fingerprints[FINGERPRINT_VERSION] = rasa_version

        config = importer.get_config()
        self._compare_or_memorize(
            fingerprint_key=FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY,
            new_fingerprint=self._get_fingerprint_of_config_without_epochs(config),
            error_message=(
                "Cannot finetune because more than just the 'epoch' keys have been "
                "changed in the configuration. "
                "Please revert your configuration and only change "
                "the 'epoch' settings where needed."
            ),
        )

        if core:
            # NOTE: If there's a consistency check between domain and core training data
            # that ensures domain and core training data are consistent, then we can
            # drop this check.
            domain = importer.get_domain()
            self._compare_or_memorize(
                fingerprint_key=FINGERPRINT_CORE,
                new_fingerprint=self._get_fingerprint_of_domain_without_responses(
                    domain
                ),
                error_message=(
                    "Cannot finetune because more than just the responses have been "
                    "changed in the domain."
                    "Please revert all settings in your domain file (except the "
                    "'responses')."
                ),
            )

        if nlu:
            nlu_data = importer.get_nlu_data()
            self._compare_or_memorize(
                fingerprint_key=FINGERPRINT_NLU,
                new_fingerprint=nlu_data.label_fingerprint(),
                error_message=(
                    "Cannot finetune because NLU training data contains new labels "
                    "or does not contain any examples for some known labels. "
                    "Please make sure that the NLU data that you use "
                    "for finetuning contains at least one example for every label "
                    "(i.e. intent, action name, ...) that was included in the NLU "
                    "data used for training the model which we attempt to finetune "
                    "now. Moreover, you must not add labels that were not included "
                    "during training before. "
                ),
            )

        self.persist()

    def _compare_or_memorize(
        self, fingerprint_key: Text, new_fingerprint: Text, error_message: Text,
    ) -> None:
        """Compares given fingerprint if we are finetuning, otherwise just saves it.

        Args:
           fingerprint_key: name of the fingerprint
           new_fingerprint: a new fingerprint value
           error_message: message of `InvalidConfigException` that will be raised if
              a fingerprint is stored under `fingerprint_key` and differs from the
              `new_fingerprint` - and we're in finetuning mode (according to the
              execution context of this component)

        Raises:
           `InvalidConfigException` if and old fingerprint exists and differs from
           the new one
        """
        if self._is_finetuning:
            old_fingerprint = self._fingerprints[fingerprint_key]
            if old_fingerprint != new_fingerprint:
                raise InvalidConfigException(error_message)
        else:
            self._fingerprints[fingerprint_key] = new_fingerprint

    @staticmethod
    def _get_fingerprint_of_domain_without_responses(domain: Domain) -> Text:
        """Returns a fingerprint of a version of the given domain without responses.

        Args:
            domain: a domain
        Returns:
            fingerprint
        """
        domain = copy.copy(domain)
        domain.responses = {}
        return domain.fingerprint()

    @staticmethod
    def _get_fingerprint_of_config_without_epochs(
        config: Optional[Dict[Text, Any]],
    ) -> Text:
        """Returns a fingerprint of the given configurations with "epoch" keys removed.

        Note that the epoch keys are only removed if they are sub-keys of a
        "pipeline" or "policies" key.

        Args:
            config: a configuration
        Returns:
            fingerprint
        """
        if not config:
            return ""

        copied_config = copy.deepcopy(config)

        for key in ["pipeline", "policies"]:
            if copied_config.get(key):
                for p in copied_config[key]:
                    if "epochs" in p:
                        del p["epochs"]

        return rasa.shared.utils.io.deep_container_fingerprint(copied_config)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> FineTuningValidator:
        """Creates a new `FineTuningValidator` (see parent class for full docstring)."""
        return cls(
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )

    def persist(self) -> None:
        """Persists this `FineTuningValidator`."""
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                filename=path / self.FILENAME, obj=self._fingerprints
            )

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GraphComponent:
        """Loads a `FineTuningValidator` (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as path:
                fingerprints = rasa.shared.utils.io.read_json_file(
                    filename=path / cls.FILENAME,
                )
                return cls(
                    model_storage=model_storage,
                    execution_context=execution_context,
                    resource=resource,
                    fingerprints=fingerprints,
                )
        except ValueError as e:
            raise InvalidConfigException(
                f"Loading {cls.__name__} failed. Ensure that the {cls.__name__} "
                f"is part of your training graph and re-train your models before "
                f"attempting to use the {cls.__name__}."
            ) from e
