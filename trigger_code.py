import requests

# The IP addresses of your two NodeMCUs (Change these to your actual IPs)
AC_IP = "http://192.168.0.4"
RELAY_IP = "http://192.168.0.X" # Whatever IP your relay board got

def control_ac(state):
    try:
        if state == "on":
            requests.get(f"{AC_IP}/ac/on", timeout=5)
            print("Jarvis: AC is now ON.")
        elif state == "off":
            requests.get(f"{AC_IP}/ac/off", timeout=5)
            print("Jarvis: AC is now OFF.")
    except Exception as e:
        print(f"Jarvis: I couldn't reach the AC controller. Error: {e}")

def control_light(state):
    try:
        if state == "on":
            requests.get(f"{RELAY_IP}/light/on", timeout=5)
            print("Jarvis: Lights are now ON.")
        elif state == "off":
            requests.get(f"{RELAY_IP}/light/off", timeout=5)
            print("Jarvis: Lights are now OFF.")
    except Exception as e:
        print(f"Jarvis: I couldn't reach the light controller. Error: {e}")

# Test it out!
control_ac("on")
