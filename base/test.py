import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    if msg.topic == "camera/image":
        with open("capture.jpg", "wb") as f:
            f.write(msg.payload)

client = mqtt.Client()
client.on_message = on_message
client.connect("192.168.1.100", 1883)
client.subscribe("camera/capture")  # to trigger
client.subscribe("camera/image")    # to receive
client.loop_forever()