import random
import time
import json
from datetime import datetime

class SensorSimulator:
    def __init__(self):
        self.sensors = {
            "soil_moisture": {"min": 0, "max": 100},   # % moisture
            "temperature": {"min": -10, "max": 50},    # °C
            "humidity": {"min": 0, "max": 100},        # % humidity
            "light_intensity": {"min": 0, "max": 1000},# lux
            "rain_detected": {"options": [0, 1]}      # binary
        }

    def read_sensors(self):
        """Simulate reading from all sensors"""
        return {
            "timestamp": datetime.now().isoformat(),
            "soil_moisture": round(random.uniform(self.sensors["soil_moisture"]["min"],
                                                  self.sensors["soil_moisture"]["max"]), 2),
            "temperature": round(random.uniform(self.sensors["temperature"]["min"],
                                                self.sensors["temperature"]["max"]), 2),
            "humidity": round(random.uniform(self.sensors["humidity"]["min"],
                                             self.sensors["humidity"]["max"]), 2),
            "light_intensity": round(random.uniform(self.sensors["light_intensity"]["min"],
                                                    self.sensors["light_intensity"]["max"]), 2),
            "rain_detected": random.choice(self.sensors["rain_detected"]["options"])
        }

if __name__ == "__main__":
    sim = SensorSimulator()
    while True:
        data = sim.read_sensors()
        print(json.dumps(data, indent=2))
        time.sleep(5)  # Read every 5 seconds