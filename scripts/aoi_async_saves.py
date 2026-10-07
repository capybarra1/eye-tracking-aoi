"""Publish annotation exports from committed snapshots without blocking interaction.

Call request/status under the service's session lock.  wait/close must be called
without that lock: neither the worker nor its frozen session touches a live
Session, VideoSource, cursor, or tracker.
"""
from __future__ import annotations

import copy
import json
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


class _Snapshot:
    def __init__(self, session):
        self.folder = Path(session.folder).resolve()
        self.manifest = copy.deepcopy(session.manifest)
        self.template = copy.deepcopy(session.template)
        self.segment = session.segment
        self.cursor = session.cursor
        self.source = SimpleNamespace(path=Path(session.source.path),
                                      size=tuple(session.source.size),
                                      times=list(session.source.times))
        self.db = sqlite3.connect(':memory:', check_same_thread=False)
        try:
            session.db.commit()
            session.db.backup(self.db)
        except Exception:
            self.db.close()
            raise

    def _get(self, key):
        row = self.db.execute('SELECT value FROM kv WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _set(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)',
                        (key, json.dumps(value)))


@dataclass
class _Job:
    key: str
    first_generation: int
    generation: int
    session: _Snapshot
    database: Path
    reason: str


class AnnotationSaves:
    """One export worker; one running job and only the newest pending per folder.

    A request returns a generation.  ``wait(session, generation)`` waits for
    that snapshot, or a newer snapshot that replaced it before export started.
    Later failures cannot be mistaken for an earlier successful manual save,
    and a later retry cannot hide the failure of an explicitly awaited save.
    """
    def __init__(self, manager):
        self.manager = manager
        self._condition = threading.Condition()
        self._pending = OrderedDict()
        self._generation = {}
        self._saved = {}
        self._state = {}
        # Small completion summaries are kept so generation-specific wait stays
        # deterministic even when an export finishes before its HTTP waiter runs.
        self._completed = {}
        self._closing = False
        self._worker = threading.Thread(target=self._run, name='aoi-export', daemon=True)
        self._worker.start()

    @staticmethod
    def _key(session) -> str:
        return str(Path(session.folder).resolve())

    def request(self, session, reason: str = '自动保存') -> dict:
        key = self._key(session)
        with self._condition:
            if self._closing:
                raise RuntimeError('标注保存队列已经关闭')
            previous = self._saved.get(key)
        if previous is None:
            previous = self.manager.saved_annotations(session)
        frozen = _Snapshot(session)
        database = next((Path(row[2]) for row in session.db.execute('PRAGMA database_list')
                         if row[1] == 'main' and row[2]), None)
        if database is None:
            frozen.db.close()
            raise ValueError('标注工作区必须使用本地数据库')
        with self._condition:
            if self._closing:
                frozen.db.close()
                raise RuntimeError('标注保存队列已经关闭')
            generation = self._generation.get(key, 0) + 1
            self._generation[key] = generation
            self._saved.setdefault(key, previous)
            replaced = self._pending.get(key)
            first = replaced.first_generation if replaced else generation
            if replaced:
                replaced.session.db.close()
            self._pending[key] = _Job(key, first, generation, frozen, database, reason)
            state = {**self._saved[key], 'status': 'saving',
                     'reason': reason, 'generation': generation}
            state.pop('error', None)
            self._state[key] = state
            self._condition.notify_all()
            return dict(state)

    def status(self, session) -> dict:
        key = self._key(session)
        with self._condition:
            state = self._state.get(key)
            if state is not None:
                return dict(state)
        return self.manager.saved_annotations(session)

    def _wait_key(self, key: str, target: int) -> dict:
        with self._condition:
            while True:
                for first, last, result, error in self._completed.get(key, ()):
                    if first <= target <= last:
                        if error is not None:
                            raise error
                        return dict(result)
                self._condition.wait()

    def wait(self, session, generation: int | None = None) -> dict:
        key = self._key(session)
        with self._condition:
            target = self._generation.get(key, 0) if generation is None else generation
            if target < 0 or target > self._generation.get(key, 0):
                raise ValueError('未知的标注保存版本')
        if target:
            return self._wait_key(key, target)
        return self.manager.saved_annotations(session)

    def raise_if_error(self) -> None:
        """Fail fast if the latest requested export of any project has failed."""
        with self._condition:
            for key, state in self._state.items():
                if state['status'] == 'error':
                    # An error state always describes this folder's most recent
                    # completed generation. A new request changes it to saving.
                    error = self._completed[key][-1][3]
                    if error is not None:
                        raise error

    def wait_all(self) -> dict:
        """Await the latest generations at entry, including switched projects.

        Call outside the service lock. Requests arriving later do not extend the
        target set, except when coalescing replaces a not-yet-started snapshot.
        Finish every captured target before surfacing its first error.
        """
        with self._condition:
            targets = dict(self._generation)
        result = {}
        first_error = None
        for key, generation in targets.items():
            try:
                result[key] = self._wait_key(key, generation)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return result

    def close(self) -> None:
        """Finish the current job and each folder's newest snapshot before exit."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._worker.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closing:
                    self._condition.wait()
                if not self._pending:
                    return
                _, job = self._pending.popitem(last=False)
            result = None
            error = None
            try:
                result = self.manager.save_annotations(job.session, job.reason)
                # Own connection: a project switch may already have closed the
                # original Session. Only this metadata key is ever changed.
                with sqlite3.connect(job.database, timeout=30) as db:
                    db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)',
                               ('annotation_save', json.dumps(result)))
            except Exception as exc:
                message = '存档和数据表更新失败，请点击保存标注重试：' + str(exc)
                try:
                    error = type(exc)(message)
                except Exception:
                    error = RuntimeError(message)
                error.__cause__ = exc
            finally:
                job.session.db.close()
            with self._condition:
                self._completed.setdefault(job.key, []).append(
                    (job.first_generation, job.generation, result, error))
                if error is None:
                    self._saved[job.key] = result
                if self._generation[job.key] == job.generation:
                    self._state[job.key] = {
                        **self._saved[job.key], 'generation': job.generation,
                        'reason': job.reason, 'status': 'error' if error else 'saved'}
                    if error:
                        self._state[job.key]['error'] = str(error)
                else:
                    # A newer request still needs publishing; include any now
                    # available filenames without advertising that request saved.
                    latest = self._state[job.key]
                    self._state[job.key] = {
                        **self._saved[job.key], 'status': 'saving',
                        'reason': latest['reason'], 'generation': latest['generation']}
                self._condition.notify_all()
