#!/usr/bin/env python3
"""Thread-safe hand-off between the GUI thread and the ROS node (no ROS, no tkinter)."""
import threading
import time
from dataclasses import replace

from sobits_intball2_gnc.teleop.state import KeyState, TeleopState


class TeleopLink:
    """The GUI calls ``set_key`` / ``get_state``; the node calls ``take_key`` / ``set_state``.

    ``take_key`` reports released keys (and no enable) when the GUI has not updated within
    ``key_timeout`` seconds, so a frozen or closed GUI can never leave a key held. ``estop`` is a
    one-shot: it is delivered once and then cleared.
    """

    def __init__(self, key_timeout: float = 0.5, clock=time.monotonic) -> None:
        self._lock = threading.Lock()
        self._timeout = float(key_timeout)
        self._clock = clock
        self._key = KeyState()
        self._key_t = None
        self._estop = False
        self._state = TeleopState()

    def set_key(self, key: KeyState) -> None:
        with self._lock:
            self._estop = self._estop or key.estop
            self._key = replace(key, estop=False)
            self._key_t = self._clock()

    def take_key(self) -> KeyState:
        with self._lock:
            estop, self._estop = self._estop, False
            fresh = self._key_t is not None and self._clock() - self._key_t <= self._timeout
            key = self._key if fresh else KeyState()
            return replace(key, estop=estop)

    def set_state(self, state: TeleopState) -> None:
        with self._lock:
            self._state = state

    def get_state(self) -> TeleopState:
        with self._lock:
            return self._state
