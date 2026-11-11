import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
TOPIC = "ME193"   # change to the topic your professor gives

def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    print(f"\n[{msg.topic}] {msg.payload.decode()}\n> ", end="", flush=True)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_start()

print(f"Sending to '{TOPIC}' on {BROKER}. Type a message and press Enter (Ctrl+C to quit).")
try:
    while True:
        message = input("> ")
        if message:
            client.publish(TOPIC, message).wait_for_publish()
            print("sent")
except (KeyboardInterrupt, EOFError):
    pass
finally:
    client.loop_stop()
    client.disconnect()
