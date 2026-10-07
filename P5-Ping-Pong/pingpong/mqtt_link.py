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
