"""Wire a paho client to a ScorePublisher: LWT on the status topic, backoff, async connect.

The game must boot offline (connect_async + loop_start), a reconnect republishes
the score via publisher.on_connect(), and the last-will lives ONLY on the status
topic -- a will on the score topic could put a non-float in front of the grader.
"""

import uuid

import config


def make_paho_client():
    import paho.mqtt.client as mqtt

    return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pp-rafae-" + uuid.uuid4().hex[:8])


def attach(client, publisher, *, host=None, port=None, keepalive=None, status_topic=None):
    client.will_set(status_topic or config.STATUS_TOPIC, "offline", qos=1, retain=True)
    client.reconnect_delay_set(1, 30)
    client.on_connect = lambda c, userdata, flags, reason_code, properties=None: publisher.on_connect()
    client.connect_async(host or config.BROKER_HOST, port or config.BROKER_PORT,
                         keepalive or config.KEEPALIVE_S)
    client.loop_start()
    return client


def status_fn(client):
    """HUD text from the real connection state: "ok" while connected, else "offline"."""
    def status():
        is_connected = getattr(client, "is_connected", None)
        return "ok" if is_connected is not None and is_connected() else "offline"
    return status


def _announce_offline(client, publisher):
    if publisher.no_publish:
        return
    info = client.publish(publisher.status_topic, "offline", qos=1, retain=True)
    wait = getattr(info, "wait_for_publish", None)
    if wait is not None:
        wait(timeout=2.0)                      # queued messages go out in order: the score first


def shutdown(client, publisher):
    """Clean exit: last value, a flushed "offline" status, then hang up and stop the network loop.

    Each step has its own try/except so one failure (broker gone, loop already stopped) never
    stops the rest.  The score topic is never written here except by publisher.close().
    """
    for step in (publisher.close, lambda: _announce_offline(client, publisher), client.disconnect,
                 client.loop_stop):
        try:
            step()
        except Exception:
            pass
