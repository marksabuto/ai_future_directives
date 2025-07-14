# 🌾 SmartAgriSense – AI-Powered Smart Agriculture System

## 🎯 Objective

To design an **IoT-enabled smart agriculture system** that uses **real-time environmental data** and **AI models** to monitor crop conditions, optimize irrigation/fertilization, and predict yields.

## 🔧 Key Components

### Sensors:
- Soil Moisture Sensor
- Temperature & Humidity (DHT22)
- Light Intensity (BH1750)
- Rain Detection Sensor
- GPS Module

### AI Model:
- Random Forest Regressor trained on historical and real-time sensor data
- Predicts crop yield based on environmental conditions

### Data Flow:
- Sensors → Microcontroller (e.g., ESP32/Raspberry Pi) → Local AI Processing / Cloud Upload → Prediction Engine → Farmer Dashboard

## 💡 Benefits

- Increases productivity by optimizing water and nutrient usage
- Reduces waste and improves sustainability
- Enhances resilience against climate change
- Enables precision farming at scale

## 🧠 Edge AI Integration

Edge computing allows for real-time anomaly detection (e.g., sudden soil dryness or temperature spikes), reducing dependency on cloud services and ensuring offline operation in remote areas.

## 📊 Deliverables

| File | Description |
|------|-------------|
| `proposal.md` | This file – project overview |
| `sensor_simulation.py` | Simulates sensor readings |
| `ai_model.py` | Trains and runs prediction model |
| `diagrams/data_flow.drawio.png` | Data flow visualization |

---

Made with ❤️ by Marks