import time, json, psutil, paho.mqtt.client as mqtt

BROKER='mqtt'
client = mqtt.Client()
client.connect(BROKER, 1883, 60)

while True:
    payload = {
        'cpu': psutil.cpu_percent(),
        'mem': psutil.virtual_memory().percent,
        'tempC': 55.0,  # TODO: read Pi sensor
    }
    client.publish('reachy/telemetry/health', json.dumps(payload), qos=0, retain=False)
    time.sleep(2)