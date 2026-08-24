#!/usr/bin/env python3
"""OS-level singleton lock to prevent duplicate ``control_node`` launches.

Uses ``flock`` on a fixed lock file: it is kernel-held and process-scoped, so
it is released automatically on process exit (normal or crash) with no stale
lock file to clean up (docs/main_plan.md, control_node multi-launch incident).
Not effective across separate machines/containers -- this project runs a
single control_node per container, so that is out of scope.
"""
import fcntl

DEFAULT_LOCK_PATH = "/tmp/intball2_control_node.lock"


class SingletonLockError(RuntimeError):
    """Raised when another process already holds the singleton lock."""


def acquire_singleton_lock(path: str = DEFAULT_LOCK_PATH):
    """Acquire an exclusive, non-blocking lock on ``path``.

    Returns the open file object; keep it referenced for the process
    lifetime (closing it, including via garbage collection, releases the
    lock). Raises :class:`SingletonLockError` if another process holds it.
    """
    lock_file = open(path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        raise SingletonLockError(
            "another process already holds the lock at %r" % path
        )
    return lock_file
