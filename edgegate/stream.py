"""Frames from the camera host to a device's live runner, over TCP.

One message per frame: a fixed header (seq, session time t, capture wall time on the
camera host, JPEG size), then the JPEG bytes. They are the bytes the session
recording keeps, so a device sees exactly what the recorded-session measurement saw.

Copied to the devices with scripts/live_device.py: standard library only, Python 3.10.
"""
import queue
import socket
import struct
import threading
import time

HEADER = struct.Struct(">IddI")  # seq, t, wall, size


class Sender:
    """Streams frames to one device in a thread. put() never blocks the camera: a frame
    is dropped (and counted) when the device falls behind or is not connected, and the
    sender reconnects every second."""

    def __init__(self, host, port, depth=30):
        self.addr = (host, int(port))
        self.q = queue.Queue(depth)
        self.sent = self.dropped = 0
        self.connected = False
        threading.Thread(target=self._run, daemon=True).start()

    def put(self, seq, t, wall, jpeg):
        try:
            self.q.put_nowait((seq, t, wall, jpeg))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            try:
                with socket.create_connection(self.addr, timeout=5) as s:
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.settimeout(None)
                    self.connected = True
                    while True:
                        seq, t, wall, jpeg = self.q.get()
                        try:
                            s.sendall(HEADER.pack(seq, t, wall, len(jpeg)) + jpeg)
                            self.sent += 1
                        finally:
                            self.q.task_done()
            except OSError:
                self.connected = False
                time.sleep(1.0)

    def flush(self, timeout=10.0):
        """Wait until every queued frame has gone out, or the timeout."""
        end = time.monotonic() + timeout
        while self.q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.05)


def receive(conn):
    """Yield (seq, t, wall, jpeg) from one connection until it closes."""
    f = conn.makefile("rb")
    while True:
        head = f.read(HEADER.size)
        if len(head) < HEADER.size:
            return
        seq, t, wall, size = HEADER.unpack(head)
        jpeg = f.read(size)
        if len(jpeg) < size:
            return
        yield seq, t, wall, jpeg


def target(spec):
    """HOST:PORT -> (host, port)."""
    host, _, port = spec.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"need HOST:PORT, got {spec!r}")
    return host, int(port)
