"""Private, single-owner local request journal. Never resumes model work."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3


class ReplayStore:
    def __init__(self, output: Path):
        self.connection = None
        self.lock = None
        if os.name != 'posix':
            raise RuntimeError('Durable local serving currently requires POSIX file locking')
        import fcntl
        try:
            self.lock = os.open(output / 'replay.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = output / 'replay.sqlite3'
            if path.is_symlink() or any((output / ('replay.sqlite3' + suffix)).is_symlink()
                                       for suffix in ('-journal', '-wal', '-shm')):
                raise ValueError('Replay files must not be symlinks')
            if not path.exists() and next(output.glob('*-started.json'), None) is not None:
                raise ValueError('Legacy attempts lack durable replay keys; preserve this directory and use a new one')
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                if os.fstat(fd).st_mode & 0o077:
                    raise ValueError('Replay database must be private (mode 600)')
            finally:
                os.close(fd)
            self.connection = sqlite3.connect(path, timeout=0)
            self.connection.execute('PRAGMA journal_mode=DELETE')
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.execute('PRAGMA cache_size=-1024')
            self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), binding TEXT NOT NULL)')
            self.connection.execute('CREATE TABLE IF NOT EXISTS attempts (key_hash TEXT PRIMARY KEY, digest TEXT NOT NULL, request_id TEXT NOT NULL, status INTEGER, body BLOB, headers TEXT, response_sha256 TEXT)')
            self.connection.commit()
        except BaseException:
            self.close()
            raise

    def bind(self, identity: dict) -> None:
        binding = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False,
                                            separators=(',', ':')).encode()).hexdigest()
        row = self.connection.execute('SELECT binding FROM metadata WHERE id=1').fetchone()
        if row is None:
            if self.count():
                raise ValueError('Replay binding is missing for existing attempts')
            with self.connection:
                self.connection.execute('INSERT INTO metadata VALUES (1, ?)', (binding,))
        elif row[0] != binding:
            raise ValueError('Replay directory belongs to a different model, key or server protocol; use a new directory')

    @staticmethod
    def key(value: str) -> str:
        return hashlib.sha256(value.encode('ascii')).hexdigest()

    def get(self, key: str):
        row = self.connection.execute('SELECT digest, request_id, status, body, headers, response_sha256 FROM attempts WHERE key_hash=?',
                                       (self.key(key),)).fetchone()
        if row is not None:
            _, _, status, body, headers, checksum = row
            if status is None:
                if any(value is not None for value in (body, headers, checksum)):
                    raise ValueError('Partial replay result')
            elif (type(status) is not int or not 100 <= status <= 599 or type(body) is not bytes
                  or len(body) > 1024 * 1024 or type(headers) is not str
                  or self.checksum(status, body, headers) != checksum):
                raise ValueError('Replay result integrity mismatch')
            return row[:5]
        return None

    @staticmethod
    def checksum(status: int, body: bytes, headers: str) -> str:
        return hashlib.sha256(str(status).encode() + b'\0' + headers.encode() + b'\0' + body).hexdigest()

    def count(self) -> int:
        return self.connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]

    def reserve(self, key: str, digest: str, request_id: str) -> None:
        # Commit before any model work. A crash from here onward is an uncertain
        # attempt, never permission to sample a second response.
        with self.connection:
            self.connection.execute('INSERT INTO attempts (key_hash,digest,request_id) VALUES (?,?,?)',
                                    (self.key(key), digest, request_id))

    def finish(self, key: str, status: int, body: bytes, headers: dict) -> None:
        if len(body) > 1024 * 1024:
            raise ValueError('Replay response exceeds one MiB')
        header_json = json.dumps(headers, sort_keys=True)
        with self.connection:
            changed = self.connection.execute('UPDATE attempts SET status=?,body=?,headers=?,response_sha256=? WHERE key_hash=? AND status IS NULL',
                (status, body, header_json, self.checksum(status, body, header_json), self.key(key)))
            if changed.rowcount != 1:
                raise ValueError('Only a reserved attempt may be completed once')

    def close(self) -> None:
        try:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
        finally:
            if self.lock is not None:
                os.close(self.lock)
                self.lock = None
