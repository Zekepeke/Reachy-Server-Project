import time, json, os, psutil, yaml
import paho.mqtt.client as mqtt

with open("edge/config.yaml","r") as f:
    CFG = yaml.safe_load(f)

BROKER = CFG["mqtt"]["broker"]
PORT   = int(CFG["mqtt"]["port"])
BASE   = CFG["mqtt"]["base_topic"]
INTERVAL = int(CFG["telemetry"]["interval_sec"])

user_env = CFG["security"].get("mqtt_user_env","")
pass_env = CFG["security"].get("mqtt_pass_env","")
USER = os.getenv(user_env) if user_env else None
PWD  = os.getenv(pass_env) if pass_env else None

client = mqtt.Client()
if USER and PWD:
    client.username_pw_set(USER, PWD)
client.connect(BROKER, PORT, 60)

def read_cpu_temp_c():
    path = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(path) as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None

while True:
    payload = {
        "cpu_percent": psutil.cpu_percent(),
        "mem_percent": psutil.virtual_memory().percent,
        "tempC": read_cpu_temp_c(),
        "uptime_sec": int(time.time() - psutil.boot_time()),
    }
    client.publish(f"{BASE}/telemetry/health", json.dumps(payload), qos=0, retain=False)
    time.sleep(INTERVAL)