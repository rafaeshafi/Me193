"""MQTT chat: connect to a broker, subscribe to topics, watch and send messages.

A small window for talking to the robots and to each other:

  - connect to any broker (defaults from config.py)
  - subscribe to as many topics as you like (wildcards work: ME193/#)
  - see every message live, with time and topic
  - send to any topic; one-click buttons for start / caught / goal
  - optionally put your name in front of chat lines

Messages are sent as plain text, exactly as typed, so the robot programs
understand them ("start" really is "start"). Turn on "add my name" only for
human chat, because "Rafae: start" is not a start command.

    python mqtt_chat.py
"""

import queue
import time
import tkinter as tk
import uuid
from tkinter import scrolledtext, ttk

import paho.mqtt.client as mqtt

import config

POLL_MS = 100  # how often the window collects messages from the MQTT thread


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.inbox = queue.Queue()  # paho's network thread -> the Tk thread
        self.client = None
        self.subscribed = []
        self.sent = []  # (topic, text) we published, to recognise their echoes
        root.title("MQTT Chat")
        root.geometry("820x620")
        root.minsize(640, 480)

        # --- connection row
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Broker").pack(side="left")
        self.broker = tk.StringVar(value=config.BROKER)
        ttk.Entry(top, textvariable=self.broker, width=22).pack(side="left", padx=4)
        ttk.Label(top, text="Port").pack(side="left")
        self.port = tk.StringVar(value=str(config.PORT))
        ttk.Entry(top, textvariable=self.port, width=6).pack(side="left", padx=4)
        ttk.Label(top, text="Name").pack(side="left", padx=(8, 0))
        self.name = tk.StringVar(value="me")
        ttk.Entry(top, textvariable=self.name, width=10).pack(side="left", padx=4)
        self.connect_btn = ttk.Button(top, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side="left", padx=6)
        self.status = ttk.Label(top, text="disconnected", foreground="gray")
        self.status.pack(side="left")

        body = ttk.Frame(root, padding=(8, 0))
        body.pack(fill="both", expand=True)

        # --- topics column
        side = ttk.Frame(body)
        side.pack(side="left", fill="y", padx=(0, 8))
        ttk.Label(side, text="Subscribed topics").pack(anchor="w")
        self.topic_list = tk.Listbox(side, width=28, height=12, exportselection=False)
        self.topic_list.pack(fill="y", expand=True)
        self.new_topic = tk.StringVar(value="ME193/#")
        entry = ttk.Entry(side, textvariable=self.new_topic)
        entry.pack(fill="x", pady=(6, 2))
        entry.bind("<Return>", lambda e: self.subscribe(self.new_topic.get()))
        buttons = ttk.Frame(side)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Subscribe",
                   command=lambda: self.subscribe(self.new_topic.get())).pack(side="left")
        ttk.Button(buttons, text="Unsubscribe", command=self.unsubscribe).pack(side="left")
        self.hide_heartbeats = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="hide robot heartbeats",
                        variable=self.hide_heartbeats).pack(anchor="w", pady=(8, 0))
        ttk.Button(side, text="Clear log", command=self.clear).pack(anchor="w", pady=4)

        # --- message log
        self.log = scrolledtext.ScrolledText(body, state="disabled", wrap="word",
                                             font="TkFixedFont", height=20)
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_config("time", foreground="gray")
        self.log.tag_config("topic", foreground="#1f6feb")
        self.log.tag_config("mine", foreground="#1a7f37")
        self.log.tag_config("info", foreground="#9a6700")

        # --- quick game messages
        quick = ttk.Frame(root, padding=(8, 6, 8, 0))
        quick.pack(fill="x")
        ttk.Label(quick, text=f"Game ({config.TOPIC}):").pack(side="left")
        for label, text in (("start", config.MSG_START), ("caught", config.MSG_CAUGHT),
                            ("goal", config.MSG_GOAL)):
            ttk.Button(quick, text=f"send '{text}'",
                       command=lambda t=text: self.publish(config.TOPIC, t)).pack(
                side="left", padx=3)

        # --- send row
        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill="x")
        ttk.Label(bottom, text="To").pack(side="left")
        self.send_topic = ttk.Combobox(bottom, width=26,
                                       values=[config.TOPIC, config.TEAM_TOPIC])
        self.send_topic.set(config.TOPIC)
        self.send_topic.pack(side="left", padx=4)
        self.message = tk.StringVar()
        msg_entry = ttk.Entry(bottom, textvariable=self.message)
        msg_entry.pack(side="left", fill="x", expand=True, padx=4)
        msg_entry.bind("<Return>", lambda e: self.send())
        self.add_name = tk.BooleanVar(value=False)
        ttk.Checkbutton(bottom, text="add my name", variable=self.add_name).pack(side="left")
        ttk.Button(bottom, text="Send", command=self.send).pack(side="left", padx=4)
        msg_entry.focus_set()

        for topic in (config.TOPIC, config.TEAM_TOPIC):
            self.add_topic(topic)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.after(POLL_MS, self.poll)

    # --- MQTT (callbacks run on paho's thread: only touch the queue there)

    def toggle_connection(self):
        if self.client is not None:
            self.disconnect()
            return
        try:
            port = int(self.port.get())
        except ValueError:
            self.note("Port must be a number.")
            return
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id=f"me193-chat-{uuid.uuid4().hex[:8]}")
        client.on_connect = lambda c, u, f, rc, p: self.inbox.put(("connected", rc))
        client.on_disconnect = lambda c, u, f, rc, p: self.inbox.put(("disconnected", rc))
        client.on_message = lambda c, u, msg: self.inbox.put(
            ("message", msg.topic, msg.payload.decode(errors="replace")))
        self.status.config(text="connecting...", foreground="#9a6700")
        try:
            client.connect_async(self.broker.get().strip(), port)
            client.loop_start()
        except (OSError, ValueError) as e:
            self.status.config(text="disconnected", foreground="gray")
            self.note(f"Could not connect: {e}")
            return
        self.client = client
        self.connect_btn.config(text="Disconnect")

    def disconnect(self):
        client, self.client = self.client, None
        client.loop_stop()
        client.disconnect()
        self.connect_btn.config(text="Connect")
        self.status.config(text="disconnected", foreground="gray")
        self.note("Disconnected.")

    def add_topic(self, topic):
        if topic and topic not in self.subscribed:
            self.subscribed.append(topic)
            self.topic_list.insert("end", topic)
            values = list(self.send_topic["values"])
            if topic not in values and "#" not in topic and "+" not in topic:
                self.send_topic["values"] = values + [topic]
            return True
        return False

    def subscribe(self, topic):
        topic = topic.strip()
        if not topic:
            return
        if self.add_topic(topic) and self.client is not None:
            self.client.subscribe(topic)
            self.note(f"Subscribed to {topic}")

    def unsubscribe(self):
        selection = self.topic_list.curselection()
        if not selection:
            self.note("Pick a topic in the list first.")
            return
        topic = self.subscribed.pop(selection[0])
        self.topic_list.delete(selection[0])
        if self.client is not None:
            self.client.unsubscribe(topic)
        self.note(f"Unsubscribed from {topic}")

    def publish(self, topic, text):
        if self.client is None:
            self.note("Connect first.")
            return
        self.client.publish(topic, text)
        if any(mqtt.topic_matches_sub(sub, topic) for sub in self.subscribed):
            self.sent.append((topic, text))  # shown when the broker echoes it back
        else:
            self.show(topic, text, mine=True)  # no echo coming: show it ourselves

    def send(self):
        text = self.message.get().strip()
        topic = self.send_topic.get().strip()
        if not text or not topic:
            return
        if self.add_name.get():
            text = f"{self.name.get().strip() or 'me'}: {text}"
        self.publish(topic, text)
        self.message.set("")

    # --- Tk side

    def poll(self):
        while not self.inbox.empty():
            event = self.inbox.get()
            if event[0] == "connected":
                if event[1] == 0:
                    self.status.config(text=f"connected to {self.broker.get()}",
                                       foreground="#1a7f37")
                    for topic in self.subscribed:
                        self.client.subscribe(topic)
                    self.note(f"Connected. Listening on: {', '.join(self.subscribed)}")
                else:
                    self.note(f"Broker refused the connection ({event[1]}).")
            elif event[0] == "disconnected" and self.client is not None:
                self.status.config(text="reconnecting...", foreground="#9a6700")
            elif event[0] == "message":
                _, topic, text = event
                if self.hide_heartbeats.get() and topic == config.TEAM_TOPIC and \
                        text.split(" ", 1)[0] in (config.STATE_CMD, config.GLOVE_CMD):
                    continue
                mine = (topic, text) in self.sent
                if mine:
                    self.sent.remove((topic, text))
                self.show(topic, text, mine)
        self.root.after(POLL_MS, self.poll)

    def write(self, *parts):
        self.log.config(state="normal")
        for text, tag in parts:
            self.log.insert("end", text, tag)
        self.log.insert("end", "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def show(self, topic, text, mine):
        self.write((time.strftime("%H:%M:%S "), "time"), (f"[{topic}] ", "topic"),
                   (text, "mine" if mine else ""))

    def note(self, text):
        self.write((time.strftime("%H:%M:%S "), "time"), (f"* {text}", "info"))

    def clear(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def quit(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()
