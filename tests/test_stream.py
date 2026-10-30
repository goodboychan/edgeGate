import itertools
import socket
import threading
import unittest

from edgegate import stream


class StreamTest(unittest.TestCase):
    def test_frames_arrive_in_order_with_their_times_and_bytes(self):
        srv = socket.create_server(("127.0.0.1", 0))
        got = []

        def serve():
            conn, _ = srv.accept()
            with conn:
                got.extend(itertools.islice(stream.receive(conn), 20))

        th = threading.Thread(target=serve)
        th.start()
        frames = [(i, i / 15, 1000.0 + i, bytes([i]) * (100 + i)) for i in range(20)]
        sender = stream.Sender("127.0.0.1", srv.getsockname()[1])
        for f in frames:  # queued until the connection is up
            sender.put(*f)
        sender.flush()
        th.join(5)
        srv.close()
        self.assertEqual(got, frames)
        self.assertEqual((sender.sent, sender.dropped), (20, 0))

    def test_target(self):
        self.assertEqual(stream.target("192.168.50.67:8790"), ("192.168.50.67", 8790))
        with self.assertRaises(ValueError):
            stream.target("jetson")


if __name__ == "__main__":
    unittest.main()
