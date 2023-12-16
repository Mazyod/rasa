from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os

from typing import AsyncGenerator, Dict, Optional, Text, Union

from rasa.shared.exceptions import RasaException, ConnectionException
import rasa.shared.utils.common
from rasa.core.constants import DEFAULT_LOCK_LIFETIME
from rasa.core.lock import TicketLock

logger = logging.getLogger(__name__)


def _get_lock_lifetime() -> int:
    return int(os.environ.get("TICKET_LOCK_LIFETIME", 0)) or DEFAULT_LOCK_LIFETIME


LOCK_LIFETIME = _get_lock_lifetime()
DEFAULT_SOCKET_TIMEOUT_IN_SECONDS = 10

DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX = "lock:"


# noinspection PyUnresolvedReferences
class LockError(RasaException):
    """Exception that is raised when a lock cannot be acquired.

    Attributes:
         message (str): explanation of which `conversation_id` raised the error
    """

    pass


class LockStore:
    """Base class for ticket locks."""

    @staticmethod
    def create(obj: Union[LockStore, None]) -> LockStore:
        """Factory to create a lock store."""
        if isinstance(obj, LockStore):
            return obj

    @staticmethod
    def create_lock(conversation_id: Text) -> TicketLock:
        """Create a new `TicketLock` for `conversation_id`."""
        return TicketLock(conversation_id)

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """Fetch lock for `conversation_id` from storage."""
        raise NotImplementedError

    def delete_lock(self, conversation_id: Text) -> None:
        """Delete lock for `conversation_id` from storage."""
        raise NotImplementedError

    def save_lock(self, lock: TicketLock) -> None:
        """Commit `lock` to storage."""
        raise NotImplementedError

    def issue_ticket(
        self, conversation_id: Text, lock_lifetime: float = LOCK_LIFETIME
    ) -> int:
        """Issue new ticket with `lock_lifetime` for lock associated with
        `conversation_id`.

        Creates a new lock if none is found.
        """
        logger.debug(f"Issuing ticket for conversation '{conversation_id}'.")
        try:
            lock = self.get_or_create_lock(conversation_id)
            ticket = lock.issue_ticket(lock_lifetime)
            self.save_lock(lock)

            return ticket
        except Exception as e:
            raise LockError(f"Error while acquiring lock. Error:\n{e}")

    @asynccontextmanager
    async def lock(
        self,
        conversation_id: Text,
        lock_lifetime: float = LOCK_LIFETIME,
        wait_time_in_seconds: float = 1,
    ) -> AsyncGenerator[TicketLock, None]:
        """Acquire lock with lifetime `lock_lifetime`for `conversation_id`.

        Try acquiring lock with a wait time of `wait_time_in_seconds` seconds
        between attempts. Raise a `LockError` if lock has expired.
        """
        ticket = self.issue_ticket(conversation_id, lock_lifetime)
        try:

            yield await self._acquire_lock(
                conversation_id, ticket, wait_time_in_seconds
            )
        finally:
            self.cleanup(conversation_id, ticket)

    async def _acquire_lock(
        self, conversation_id: Text, ticket: int, wait_time_in_seconds: float
    ) -> TicketLock:
        logger.debug(f"Acquiring lock for conversation '{conversation_id}'.")
        while True:
            # fetch lock in every iteration because lock might no longer exist
            lock = self.get_lock(conversation_id)

            # exit loop if lock does not exist anymore (expired)
            if not lock:
                break

            # acquire lock if it isn't locked
            if not lock.is_locked(ticket):
                logger.debug(f"Acquired lock for conversation '{conversation_id}'.")
                return lock

            items_before_this = ticket - (lock.now_serving or 0)

            logger.debug(
                f"Failed to acquire lock for conversation ID '{conversation_id}' "
                f"because {items_before_this} other item(s) for this "
                f"conversation ID have to be finished processing first. "
                f"Retrying in {wait_time_in_seconds} seconds ..."
            )

            # sleep and update lock
            await asyncio.sleep(wait_time_in_seconds)
            self.update_lock(conversation_id)

        raise LockError(
            f"Could not acquire lock for conversation_id '{conversation_id}'."
        )

    def update_lock(self, conversation_id: Text) -> None:
        """Fetch lock for `conversation_id`, remove expired tickets and save lock."""
        lock = self.get_lock(conversation_id)
        if lock:
            lock.remove_expired_tickets()
            self.save_lock(lock)

    def get_or_create_lock(self, conversation_id: Text) -> TicketLock:
        """Fetch existing lock for `conversation_id`.

        Alternatively, create a new one if it doesn't exist.
        """
        existing_lock = self.get_lock(conversation_id)

        if existing_lock:
            return existing_lock

        return self.create_lock(conversation_id)

    def is_someone_waiting(self, conversation_id: Text) -> bool:
        """Return whether someone is waiting for lock for this `conversation_id`."""
        lock = self.get_lock(conversation_id)
        if lock:
            return lock.is_someone_waiting()

        return False

    def finish_serving(self, conversation_id: Text, ticket_number: int) -> None:
        """Finish serving ticket with `ticket_number` for `conversation_id`.

        Removes ticket from lock and saves lock.
        """
        lock = self.get_lock(conversation_id)
        if lock:
            lock.remove_ticket_for(ticket_number)
            self.save_lock(lock)

    def cleanup(self, conversation_id: Text, ticket_number: int) -> None:
        """Remove lock for `conversation_id` if no one is waiting."""
        self.finish_serving(conversation_id, ticket_number)
        if not self.is_someone_waiting(conversation_id):
            self.delete_lock(conversation_id)

    @staticmethod
    def _log_deletion(conversation_id: Text, deletion_successful: bool) -> None:
        if deletion_successful:
            logger.debug(f"Deleted lock for conversation '{conversation_id}'.")
        else:
            logger.debug(f"Could not delete lock for conversation '{conversation_id}'.")


class InMemoryLockStore(LockStore):
    """In-memory store for ticket locks."""

    def __init__(self) -> None:
        """Initialise dictionary of locks."""
        self.conversation_locks: Dict[Text, TicketLock] = {}
        super().__init__()

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """Get lock for conversation if it exists."""
        return self.conversation_locks.get(conversation_id)

    def delete_lock(self, conversation_id: Text) -> None:
        """Delete lock for conversation."""
        deleted_lock = self.conversation_locks.pop(conversation_id, None)
        self._log_deletion(
            conversation_id, deletion_successful=deleted_lock is not None
        )

    def save_lock(self, lock: TicketLock) -> None:
        """Save lock in store."""
        self.conversation_locks[lock.conversation_id] = lock
