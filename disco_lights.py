import random
import time

import requests

RELAY_IP = "192.168.0.8"

# Use a session to keep connections alive, making the requests faster
session = requests.Session()

print("🪩 Starting Disco Mode! Press Ctrl+C to stop...")

try:
    while True:
        # Generate a random state for each of the 4 relays (1 = ON, 0 = OFF)
        states = [random.choice([0, 1]) for _ in range(4)]

        # Fire off the requests for all 4 relays at once
        for i, state in enumerate(states, start=1):
            action = "on" if state == 1 else "off"
            url = f"http://{RELAY_IP}/relay{i}/{action}"
            try:
                # Very short timeout so a laggy request doesn't ruin the beat
                session.get(url, timeout=0.5)
            except requests.exceptions.RequestException:
                pass  # Ignore errors to keep the beat going

        # Adjust the speed of the strobe (0.15 seconds is a fast strobe, 0.3 is slower)
        time.sleep(0.15)

except KeyboardInterrupt:
    # When you press Ctrl+C, gracefully exit and turn all lights off
    print("\n🛑 Disco Mode stopped. Turning all lights off...")
    for i in range(1, 5):
        url = f"http://{RELAY_IP}/relay{i}/off"
        try:
            session.get(url, timeout=1.0)
        except Exception:
            pass
    print("Lights are off. Party's over!")
