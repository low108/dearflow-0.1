"""Builds the storage pieces from config."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from dearflow.config import PersistenceConfig
from dearflow.persistence.database import MEMORY, Database
from dearflow.persistence.events import RunEventStore
from dearflow.persistence.runs import RunStore
from dearflow.persistence.threads import ThreadStore


@dataclass
class Persistence:
    """Two layers, as in DeerFlow:

    - `checkpointer`: LangGraph saves the full state (all messages) per thread.
    - `threads` / `runs`: small tables about those threads and runs.

    With persistence off, the checkpointer is in memory and the tables live in a
    throwaway in-memory SQLite, so the rest of the code never checks "is it on?".
    """

    checkpointer: BaseCheckpointSaver
    threads: ThreadStore
    runs: RunStore
    events: RunEventStore
    database: Database

    @classmethod
    def from_config(cls, config: PersistenceConfig) -> Persistence:
        if not config.enabled:
            return cls.in_memory()
        database = Database(config.path)
        # The checkpointer gets its own connection to the same file (WAL allows it),
        # because it serializes its own writes and must not interleave with ours.
        saver = SqliteSaver(sqlite3.connect(config.path, check_same_thread=False))
        return cls._assemble(saver, database)

    @classmethod
    def in_memory(cls) -> Persistence:
        return cls._assemble(InMemorySaver(), Database(MEMORY))

    @classmethod
    def _assemble(cls, saver: BaseCheckpointSaver, database: Database) -> Persistence:
        runs = RunStore(database)
        threads = ThreadStore(database)
        # A crash can leave runs "running" and threads "busy" forever; clear them at startup.
        for thread_id in runs.interrupt_unfinished():
            threads.set_status(thread_id, "idle")
        return cls(saver, threads, runs, RunEventStore(database), database)

    def close(self) -> None:
        connection = getattr(self.checkpointer, "conn", None)
        if connection is not None:
            connection.close()
        self.database.close()
