# 🚍 AI-Powered Multilingual Bus Stop Announcement System

## 📌 Overview

This project presents an **intelligent, context-aware bus stop announcement system** designed to improve public transport accessibility and passenger experience. The system simulates real-world bus movement and delivers **real-time multilingual audio announcements** in **English, Hindi, and Telugu**, adapting dynamically to environmental and operational conditions.

The solution combines **AI-driven ETA prediction**, **Neuro-Fuzzy reasoning**, and **language generation (TTS)** to demonstrate how modern AI techniques can be applied to smart transportation systems.

---

## ✨ Key Highlights

* 🌍 **Multilingual Announcements** – English, Hindi, Telugu
* 🧠 **AI-Based ETA Prediction** using Sugeno-type Neuro-Fuzzy Model
* 🌦️ **Live Weather Awareness** (wttr.in API integration)
* 🛰️ **GPS-Based Distance Calculation** using latitude & longitude
* 🚏 **Event-Based Announcements** (Approach, Arrival, Door Closing)
* ♿ **Accessibility-Oriented Design** for visually impaired passengers

---

## 🧠 AI & Intelligence Layer

The core intelligence of the system lies in its **Sugeno-type Neuro-Fuzzy Model**, implemented using **PyTorch**.

### Model Capabilities:

* Learns travel-time patterns from simulated GPS data
* Handles uncertainty using fuzzy logic
* Adapts ETA predictions based on:

  * Distance between stops
  * Vehicle speed
  * Crowd-based stop time
  * Weather conditions
  * Time-of-day and weekend factors

This hybrid approach combines **neural learning** with **fuzzy reasoning**, making it efficient and suitable for real-time transport simulations.

---

## 🔊 Multilingual Text-to-Speech System

The announcement module generates **natural-sounding audio outputs** using:

* **gTTS (Google Text-to-Speech)** for language generation
* **Pygame** for real-time audio playback

### Announcement Types:

* 🚍 *Approach Alert* – Triggered ~300 meters before the stop
* ✅ *Arrival Alert* – On reaching the stop
* 🔔 *Door Closing Alert* – Includes ETA to next stop

---

## 🛠️ Technology Stack

**Programming Language:** Python

**Core Libraries & Tools:**

* PyTorch – Neuro-Fuzzy model training & inference
* Pandas, NumPy – Data preprocessing & feature extraction
* Geopy – GPS-based distance calculation
* gTTS – Multilingual speech generation
* Pygame – Audio playback
* Requests – Live weather API integration

---

## 📂 Project Structure

```
├── bus_simulation_telugu_wttr.py   # Main simulation script
├── stops_1.csv                    # Bus stop GPS & sequence data
├── bus_fuzzy_model.pth            # Trained Neuro-Fuzzy model
├── bus_training_data_with_weather.csv
├── README.md
```

---

## ▶️ How to Run

```bash
python anouncements11.py
```

🔊 Ensure speakers or headphones are connected for audio announcements.

---

## 📈 Model Training Details

* **Model Type:** Sugeno-type Neuro-Fuzzy Model
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)
* **Epochs:** 300
* **Output:** Predicted Travel Time (ETA in seconds)

The trained model demonstrates smoother and more realistic ETA predictions compared to static distance-speed calculations.

---

## 🚀 Future Enhancements

* Integration with **real GPS hardware & Raspberry Pi**
* Replacement of gTTS with **deep-learning TTS models** (Tacotron2, VITS)
* Addition of **LLM-based conversational assistant** for passengers
* Support for more regional languages
* Cloud-based real-time bus tracking and analytics

---

## 🎯 Use Case Impact

* Improves accessibility for **visually impaired and non-local passengers**
* Promotes **inclusive and smart public transport systems**
* Demonstrates practical application of **AI + Language Technologies**

---

## 📜 Conclusion

This project showcases a real-world application of **AI-powered decision-making and language generation** in the transportation domain. By combining Neuro-Fuzzy intelligence with multilingual speech output, the system highlights how technology can enhance accessibility, efficiency, and user experience in public transport.

---

👤 **Author:** Anurag Marda
