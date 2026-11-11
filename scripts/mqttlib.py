import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883


class MQTTClient:
    """Minimal paho-mqtt wrapper: use as a context manager, subscribe with callbacks, publish strings."""

    def __init__(self, host=BROKER, port=PORT):
        self.host = host
        self.port = port
        self._callbacks = {}
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_message = self._on_message

    def __enter__(self):
        self._client.connect(self.host, self.port)
        self._client.loop_start()
        return self

    def __exit__(self, *exc):
        self._client.loop_stop()
        self._client.disconnect()

    def subscribe(self, topic, callback):
        self._callbacks[topic] = callback
        self._client.subscribe(topic)

    def publish(self, topic, message):
        self._client.publish(topic, message)

    def _on_message(self, client, userdata, msg):
        callback = self._callbacks.get(msg.topic)
        if callback:
            callback(msg.topic, msg.payload.decode())
