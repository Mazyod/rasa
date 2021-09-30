import random
from rasa.shared.exceptions import RasaException
from rasa.shared.core.trackers import DialogueStateTracker
from typing import Any, Iterable, Iterator, Text, Optional
from rasa.core.tracker_store import TrackerStore
import rasa.shared.utils.io


class MarkerTrackerLoader:
    """Represents a wrapper over a `TrackerStore` with a configurable access pattern."""

    @staticmethod
    def strategy_all(count: int, keys: Iterable[Text]) -> Iterable[Text]:
        """Selects all keys from the set of keys."""
        return keys

    @staticmethod
    def strategy_first_n(count: int, keys: Iterable[Text]) -> Iterable[Text]:
        """Takes the first N keys from the set of keys."""
        return keys[:count]

    @staticmethod
    def strategy_sample(count: int, keys: Iterable[Text]) -> Iterable[Text]:
        """Takes a sample of N keys from the set of keys."""
        return random.choices(keys, k=count)

    _STRATEGY_MAP = {
        "all": strategy_all,
        "first_n": strategy_first_n,
        "sample": strategy_sample,
    }

    def __init__(
        self,
        tracker_store: TrackerStore,
        strategy: str,
        count: int = None,
        seed: Any = None,
    ) -> None:
        """Create a MarkerTrackerLoader.

        Args:
            tracker_store: The underlying tracker store to access.
            strategy: The strategy to use for selecting trackers,
                      can be 'all', 'sample', or 'first_n'.
            count: Number of trackers to return, can only be None if strategy is 'all'.
            seed: Optional seed to set up random number generator,
                  only useful if strategy is 'sample'.
        """
        self.tracker_store = tracker_store
        self.count = count

        if strategy not in MarkerTrackerLoader._STRATEGY_MAP:
            raise RasaException(
                "Invalid strategy for loading markers - '{strategy}' was given, \
                options 'all', 'sample', or 'first_n' exist"
            )

        self.strategy = MarkerTrackerLoader._STRATEGY_MAP[strategy]

        if not count:
            if strategy != "all":
                raise RasaException(
                    "Desired tracker count must be given for strategy '{strategy}'"
                )
            else:
                rasa.shared.utils.io.raise_warning(
                    "Parameter 'count' is ignored by strategy 'all'"
                )

        if seed:
            if strategy == "sample":
                random.seed(seed)
            else:
                rasa.shared.utils.io.raise_warning(
                    "Parameter 'seed' is ignored by strategy '{strategy}'"
                )

    def load(self) -> Iterator[Optional[DialogueStateTracker]]:
        """Load trackers according to strategy."""
        stored_keys = self.tracker_store.keys()
        if self.count > len(stored_keys):
            rasa.shared.utils.io.raise_warning(
                "'count' exceeds number of trackers in the store"
            )

        keys = self.strategy(self.count, stored_keys)
        for sender in keys:
            yield self.tracker_store.retrieve(sender)
