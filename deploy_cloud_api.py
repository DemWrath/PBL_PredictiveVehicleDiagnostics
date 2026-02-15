"""
DEPLOYMENT: Cloud Inference Service (Production Ready)
=======================================================

Flask API for vehicle fault detection with Gunicorn support
"""

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from diagnostic_engine import DiagnosticEngine
import json
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from lstm_classifier import AttentionLayer
except ImportError:
    print("Warning: Could not import AttentionLayer, model loading may fail")
    AttentionLayer = None

# Initialize Flask app
app = Flask(__name__)

# Global variables (loaded once at startup)
model = None
diagnostic_engine = None

# Configuration
OPTIMAL_THRESHOLD = float(os.environ.get('FAULT_THRESHOLD', '0.52'))
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Sensor normalization ranges (from training)
SENSOR_RANGES = {
    'coolant_temp': (0, 150),
    'MAP': (0, 120),
    'RPM': (0, 7000),
    'speed': (0, 250),
    'intake_temp': (0, 150),
    'MAF': (0, 300),
    'throttle': (0, 100),
    'ambient': (0, 70),
    'APP_D': (0, 100),
    'APP_E': (0, 100),
}


def load_model():
    """Load trained model at startup"""
    global model, diagnostic_engine
    
    print("="*60)
    print("VEHICLE FAULT DETECTION API - STARTING UP")
    print("="*60)
    
    # Find model file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible locations
    possible_paths = [
        os.path.join(BASE_DIR, "classifier.h5"),
        os.path.join(BASE_DIR, "models", "classifier.h5"),
        "/mnt/user-data/outputs/models/classifier.h5",
        "D:/mnt/user-data/outputs/models/classifier.h5",
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("✗ ERROR: Classifier model not found!")
        print("  Searched locations:")
        for path in possible_paths:
            print(f"    - {path}")
        raise FileNotFoundError("classifier.h5 not found")
    
    print(f"\nLoading model from: {model_path}")
    
    # Load model with custom objects
    custom_objects = {}
    if AttentionLayer is not None:
        custom_objects['AttentionLayer'] = AttentionLayer
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        print("✓ Model loaded successfully")
        
        # Print model info
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model size: {model_size_mb:.1f} MB")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise
    
    # Initialize diagnostic engine
    print("\nInitializing diagnostic engine...")
    try:
        diagnostic_engine = DiagnosticEngine()
        print("✓ Diagnostic engine ready")
    except Exception as e:
        print(f"✗ Warning: Diagnostic engine failed to load: {e}")
        print("  Predictions will work but diagnostics will be limited")
        diagnostic_engine = None
    
    print("\n" + "="*60)
    print("✓ API READY TO SERVE REQUESTS")
    print("="*60)
    print(f"  Threshold: {OPTIMAL_THRESHOLD}")
    print(f"  Port: {PORT}")
    print(f"  Debug mode: {DEBUG}")
    print("="*60 + "\n")


def normalize_sensor_data(raw_data):
    """
    Normalize sensor readings to 0-1 range
    
    Args:
        raw_data: dict with sensor readings in original units
    
    Returns:
        numpy array (1200, 10) normalized
    """
    normalized = []
    
    sensor_order = [
        'coolant_temp', 'MAP', 'RPM', 'speed', 'intake_temp',
        'MAF', 'throttle', 'ambient', 'APP_D', 'APP_E'
    ]
    
    for sensor_name in sensor_order:
        if sensor_name not in raw_data:
            raise ValueError(f"Missing sensor data: {sensor_name}")
        
        values = np.array(raw_data[sensor_name])
        
        if len(values) != 1200:
            raise ValueError(f"{sensor_name} must have exactly 1200 readings, got {len(values)}")
        
        min_val, max_val = SENSOR_RANGES[sensor_name]
        
        # MinMax normalization
        normalized_values = (values - min_val) / (max_val - min_val)
        normalized_values = np.clip(normalized_values, 0, 1)  # Safety clip
        
        normalized.append(normalized_values)
    
    # Stack to (1200, 10) shape
    sequence = np.stack(normalized, axis=1)
    
    return sequence


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API info"""
    return jsonify({
        'service': 'Vehicle Fault Detection API',
        'version': '1.0.0',
        'status': 'operational',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'batch_predict': '/batch_predict (POST)',
        },
        'model_info': {
            'threshold': OPTIMAL_THRESHOLD,
            'expected_recall': '90.5%',
            'input_length': 1200,
            'sensors': 10,
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'diagnostic_engine_ready': diagnostic_engine is not None,
        'threshold': OPTIMAL_THRESHOLD,
    })


@app.route('/predict', methods=['POST'])
def predict_fault():
    """
    Main prediction endpoint
    
    Expected input:
    {
        "vehicle_id": "VIN123456789",
        "timestamp": "2026-02-14T10:30:00Z",
        "sensor_data": {
            "coolant_temp": [90, 91, ...],  # 1200 readings
            "MAP": [45, 46, ...],
            ...
        }
    }
    
    Returns:
    {
        "vehicle_id": "VIN123456789",
        "fault_detected": true/false,
        "confidence": 0.78,
        "diagnostic": {...}
    }
    """
    try:
        # Parse request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        vehicle_id = data.get('vehicle_id', 'UNKNOWN')
        sensor_data = data.get('sensor_data')
        
        if not sensor_data:
            return jsonify({'error': 'Missing sensor_data field'}), 400
        
        # Normalize sensor data
        try:
            sequence = normalize_sensor_data(sensor_data)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Add batch dimension
        batch_sequence = np.expand_dims(sequence, axis=0)  # (1, 1200, 10)
        
        # Predict fault probability
        prediction = model.predict(batch_sequence, verbose=0)
        fault_probability = float(prediction[0][0])
        fault_detected = bool(fault_probability > OPTIMAL_THRESHOLD)
        
        # Generate diagnostic report
        if diagnostic_engine is not None:
            try:
                diagnostic = diagnostic_engine.diagnose(
                    sequence,
                    fault_probability,
                    attention_weights=None  # Not extracted for performance
                )
            except Exception as e:
                print(f"Warning: Diagnostic failed: {e}")
                diagnostic = {
                    'error': 'Diagnostic unavailable',
                    'fault_detected': fault_detected,
                }
        else:
            diagnostic = {
                'fault_detected': fault_detected,
                'message': 'Diagnostic engine not available'
            }
        
        # Build response
        response = {
            'vehicle_id': vehicle_id,
            'timestamp': data.get('timestamp'),
            'fault_detected': fault_detected,
            'confidence': fault_probability,
            'threshold_used': OPTIMAL_THRESHOLD,
            'diagnostic': diagnostic,
            'model_version': '1.0.0',
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict_fault: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple sequences
    
    Input:
    {
        "vehicle_id": "VIN123456789",
        "sequences": [
            {"timestamp": "...", "sensor_data": {...}},
            {"timestamp": "...", "sensor_data": {...}},
            ...
        ]
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        vehicle_id = data.get('vehicle_id', 'UNKNOWN')
        sequences = data.get('sequences', [])
        
        if not sequences:
            return jsonify({'error': 'No sequences provided'}), 400
        
        results = []
        
        for seq_data in sequences:
            sensor_data = seq_data.get('sensor_data')
            
            if not sensor_data:
                results.append({
                    'timestamp': seq_data.get('timestamp'),
                    'error': 'Missing sensor_data',
                    'fault_detected': None,
                })
                continue
            
            try:
                # Normalize
                sequence = normalize_sensor_data(sensor_data)
                batch_sequence = np.expand_dims(sequence, axis=0)
                
                # Predict
                prediction = model.predict(batch_sequence, verbose=0)
                fault_probability = float(prediction[0][0])
                fault_detected = bool(fault_probability > OPTIMAL_THRESHOLD)
                
                results.append({
                    'timestamp': seq_data.get('timestamp'),
                    'fault_detected': fault_detected,
                    'confidence': fault_probability,
                })
            except Exception as e:
                results.append({
                    'timestamp': seq_data.get('timestamp'),
                    'error': str(e),
                    'fault_detected': None,
                })
        
        return jsonify({
            'vehicle_id': vehicle_id,
            'results': results,
            'total_sequences': len(results),
            'successful': sum(1 for r in results if 'error' not in r),
            'faults_detected': sum(1 for r in results if r.get('fault_detected') == True),
        })
    
    except Exception as e:
        print(f"Error in batch_predict: {e}")
        return jsonify({'error': str(e)}), 500


# Initialize model on startup (not in __main__)
try:
    load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup: {e}")
    print("API will not function correctly!")


if __name__ == '__main__':
    # Development server (Flask built-in)
    print("\n⚠️  WARNING: Using Flask development server")
    print("   For production, use: gunicorn deploy_cloud_api:app\n")
    
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEBUG
    )