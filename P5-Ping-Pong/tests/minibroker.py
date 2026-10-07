"""A small in-process MQTT 3.1.1 broker, so the REAL paho client can be tested without the internet.

Enough of the protocol for this project: CONNECT (with a last will), PUBLISH (QoS 0/1, retained), SUBSCRIBE with
`+` and `#`, PINGREQ, DISCONNECT, and the will on a connection that dies.  Subscribers get messages at QoS 0 (the
SUBACK grants 0, which MQTT allows).  `drop_first=N` closes the first N connections before answering, which is what
the public test broker does now and then; `.log` lists every PUBLISH it received.
"""

import socketserver
import struct
import threading


def _read_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data


def _read_packet(sock):
    first = _read_exact(sock, 1)[0]
    length, shift = 0, 0
    while True:
        byte = _read_exact(sock, 1)[0]
        length |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return first, _read_exact(sock, length) if length else b""


def _string(body, at):
    n = struct.unpack_from(">H", body, at)[0]
    return body[at + 2:at + 2 + n].decode(), at + 2 + n


def _remaining(n):
    out = bytearray()
    while True:
        byte, n = n % 128, n // 128
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _publish_packet(topic, payload, retain=False):
    body = struct.pack(">H", len(topic.encode())) + topic.encode() + payload
    return bytes([0x30 | int(retain)]) + _remaining(len(body)) + body


def matches(flt, topic):
    f, t = flt.split("/"), topic.split("/")
    for i, part in enumerate(f):
        if part == "#":
            return True
        if i >= len(t) or (part != "+" and part != t[i]):
            return False
    return len(f) == len(t)


class MiniBroker:
    def __init__(self, drop_first=0, port=0):
        self.drop_first, self.connections = drop_first, 0
        self.retained, self.log, self._subs, self._lock = {}, [], [], threading.Lock()
        broker = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                broker._serve(self.request)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server(("127.0.0.1", port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    # --- one connection ---------------------------------------------------------------------------------------------------
    def _serve(self, sock):
        with self._lock:
            n, self.connections = self.connections, self.connections + 1
        if n < self.drop_first:
            return                                           # closed without a CONNACK
        will, clean = None, False
        try:
            first, body = _read_packet(sock)
            if first >> 4 != 1:
                return
            _, at = _string(body, 0)
            flags = body[at + 1]
            at += 4
            _, at = _string(body, at)                        # client id
            if flags & 0x04:
                topic, at = _string(body, at)
                size = struct.unpack_from(">H", body, at)[0]
                will = (topic, body[at + 2:at + 2 + size], bool(flags & 0x20))
            sock.sendall(b"\x20\x02\x00\x00")
            while True:
                first, body = _read_packet(sock)
                kind = first >> 4
                if kind == 3:
                    self._on_publish(sock, first, body)
                elif kind == 8:
                    self._on_subscribe(sock, body)
                elif kind == 12:
                    sock.sendall(b"\xd0\x00")
                elif kind == 14:
                    clean = True
                    return
        except (ConnectionError, OSError, struct.error):
            pass
        finally:
            with self._lock:
                self._subs = [(s, f) for s, f in self._subs if s is not sock]
            if will is not None and not clean:
                self._route(*will)

    def _on_publish(self, sock, first, body):
        qos, retain = (first >> 1) & 3, bool(first & 1)
        topic, at = _string(body, 0)
        pid = None
        if qos:
            pid, at = struct.unpack_from(">H", body, at)[0], at + 2
        payload = body[at:]
        self._route(topic, payload, retain)
        if pid is not None:
            sock.sendall(b"\x40\x02" + struct.pack(">H", pid))

    def _route(self, topic, payload, retain):
        with self._lock:
            self.log.append((topic, payload, retain))
            if retain:
                if payload:
                    self.retained[topic] = payload
                else:
                    self.retained.pop(topic, None)
            targets = [s for s, f in self._subs if matches(f, topic)]
        for target in targets:
            try:
                target.sendall(_publish_packet(topic, payload))
            except OSError:
                pass

    def _on_subscribe(self, sock, body):
        pid, at, granted = struct.unpack_from(">H", body, 0)[0], 2, []
        filters = []
        while at < len(body):
            flt, at = _string(body, at)
            at += 1
            filters.append(flt)
            granted.append(0)
        with self._lock:
            self._subs += [(sock, f) for f in filters]
            held = [(t, p) for t, p in self.retained.items() if any(matches(f, t) for f in filters)]
        sock.sendall(b"\x90" + _remaining(2 + len(granted)) + struct.pack(">H", pid) + bytes(granted))
        for topic, payload in held:
            sock.sendall(_publish_packet(topic, payload, retain=True))
