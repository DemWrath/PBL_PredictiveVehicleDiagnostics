# Predictive Vehicle Diagnostics

Production-deployed deep learning–based vehicle fault detection system using an LSTM model with a custom Attention mechanism.

**Live Endpoint:**
[https://pbl-predictivevehiclediagnostics.onrender.com](https://pbl-predictivevehiclediagnostics.onrender.com)

---

## Project Overview

This project provides a RESTful API for real-time vehicle fault detection using multivariate time-series sensor data.

Pipeline:

1. Accepts 1200-timestep sequences from 10 vehicle sensors
2. Performs validation and normalization
3. Runs inference using an LSTM + Attention model
4. Applies optimized decision thresholding
5. Returns structured diagnostic output

The system is deployed in a constrained 512MB cloud environment with optimized Gunicorn configuration.

---

## System Architecture

```
Sensor Data
   ↓
Validation Layer
   ↓
Normalization
   ↓
LSTM + Attention Model
   ↓
Thresholding
   ↓
Diagnostic Engine
   ↓
JSON Response
```

Design considerations:

* Single-worker Gunicorn configuration to prevent TensorFlow memory duplication
* CPU-based TensorFlow build for efficient cloud deployment
* Deterministic threshold-based decision layer
* Modular diagnostic engine abstraction

---

## Model Details

| Property           | Value                        |
| ------------------ | ---------------------------- |
| Architecture       | LSTM with Custom Attention   |
| Input Shape        | (1200 timesteps, 10 sensors) |
| Output             | Fault probability score      |
| Decision Threshold | 0.52                         |
| Model File         | `classifier.h5`              |

The attention layer enhances temporal feature weighting across long sensor sequences.

---

## Technology Stack

* Python 3.x
* Flask
* TensorFlow / Keras
* NumPy
* Gunicorn
* Render (cloud hosting)

---

## Repository Structure

```
.
├── deploy_cloud_api.py
├── classifier.h5
├── diagnostic_engine.py
├── lstm_classifier.py
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Running Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start server

```bash
python deploy_cloud_api.py
```

Server runs at:

```
http://localhost:5000
```

---

## Production Deployment

The application runs using Gunicorn:

```bash
gunicorn deploy_cloud_api:app --workers 1 --threads 2 --timeout 120
```

Single-worker configuration is required to maintain stability within a 512MB environment due to TensorFlow memory usage.

---

## API Endpoints

### Health Check

**GET** `/health`

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "diagnostic_engine_ready": true
}
```

---

### Single Prediction

**POST** `/predict`

Request:

```json
{
  "vehicle_id": "VIN123456789",
  "timestamp": "2026-02-14T10:30:00Z",
  "sensor_data": {
    "coolant_temp": [...1200 values...],
    "MAP": [...],
    "RPM": [...],
    "speed": [...],
    "intake_temp": [...],
    "MAF": [...],
    "throttle": [...],
    "ambient": [...],
    "APP_D": [...],
    "APP_E": [...]
  }
}
```

Response:

```json
{
  "vehicle_id": "VIN123456789",
  "timestamp": "2026-02-14T10:30:00Z",
  "fault_detected": true,
  "confidence": 0.78,
  "threshold_used": 0.52,
  "diagnostic": { ... },
  "model_version": "1.0.0"
}
```

---

### Batch Prediction

**POST** `/batch_predict`

Response format:

```json
{
  "vehicle_id": "VIN123456789",
  "results": [
    {
      "timestamp": "...",
      "fault_detected": false,
      "confidence": 0.21
    }
  ],
  "total_sequences": 2,
  "faults_detected": 1
}
```

---

## Input Requirements

* Exactly 10 sensor streams
* Exactly 1200 readings per sensor
* Internal normalization applied before inference

Validation errors return HTTP 400.
Unexpected internal failures return HTTP 500.

---

## Production Characteristics

* Publicly deployed inference API
* Memory-optimized configuration
* Deterministic thresholding
* Modular diagnostic abstraction layer
* Designed for scalability under higher-resource environments

---
