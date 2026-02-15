"""
LSTM VEHICLE DIAGNOSTICS - CONFIGURATION
=========================================

Central configuration for all model parameters, paths, and hyperparameters.
"""

import os

# ============================================================================
# PATHS
# ============================================================================

PATHS = {
    # Input data
    'raw_data': '/path/to/your/vehicle_data.csv',  # UPDATE THIS!
    
    # Output directories
    'output_dir': '/mnt/user-data/outputs',
    'models_dir': '/mnt/user-data/outputs/models',
    'sequences_dir': '/mnt/user-data/outputs/sequences',
    'labels_dir': '/mnt/user-data/outputs/labels',
    'logs_dir': '/mnt/user-data/outputs/logs',
    
    # Generated files
    'sequences_file': '/mnt/user-data/outputs/sequences/sequences.npz',
    'metadata_file': '/mnt/user-data/outputs/sequences/metadata.csv',
    'autoencoder_model': '/mnt/user-data/outputs/models/autoencoder.h5',
    'classifier_model': '/mnt/user-data/outputs/models/classifier.h5',
    'clean_labels': '/mnt/user-data/outputs/labels/clean_labels_v1.csv',
}

# Create directories if they don't exist
for key, path in PATHS.items():
    if key.endswith('_dir'):
        os.makedirs(path, exist_ok=True)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

DATA_CONFIG = {
    # Sequence parameters
    'sequence_length': 1200,        # 2 minutes @ 10Hz
    'stride': 600,                  # 50% overlap (1 minute)
    'sampling_rate': 10,            # Hz
    'min_sequence_length': 600,     # Discard shorter sequences
    
    # Features (sensor columns)
    'feature_columns': [
        'coolant_temp',
        'Intake Manifold Absolute Pressure [kPa]',
        'Engine RPM [RPM]',
        'Vehicle Speed Sensor [km/h]',
        'intake_temp',
        'Air Flow Rate from Mass Flow Sensor [g/s]',
        'Absolute Throttle Position [%]',
        'ambient_temp',
        'Accelerator Pedal Position D [%]',
        'Accelerator Pedal Position E [%]'
    ],
    
    # Grouping
    'trip_id_column': 'source_file',
    'time_column': 'Time',
    'label_column': 'fault',
    
    # Preprocessing
    'normalization': 'minmax',      # 'minmax', 'standard', or 'robust'
    'handle_missing': 'forward_fill',  # 'forward_fill', 'interpolate', or 'drop'
    'remove_duplicates': True,
}


# ============================================================================
# AUTOENCODER (PHASE 2 - ANOMALY DETECTION)
# ============================================================================

AUTOENCODER_CONFIG = {
    # Architecture
    'input_shape': (1200, 10),      # (sequence_length, n_features)
    'encoder_lstm_units': [128, 64],
    'latent_dim': 32,
    'decoder_lstm_units': [64, 128],
    'activation': 'tanh',
    
    # Regularization
    'dropout': 0.2,
    'recurrent_dropout': 0.0,       # Set to 0 to enable cuDNN (3-5x faster!)
    'kernel_regularizer': 'l2',
    'l2_lambda': 0.0001,
    
    # Training
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'mse',
    'validation_split': 0.15,
    
    # Early stopping
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 15,
        'restore_best_weights': True,
        'min_delta': 0.0001
    },
    
    # Learning rate reduction
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 10,
        'min_lr': 0.00001
    },
    
    # Data filtering
    'train_on_normal_only': True,   # Only fault=0 sequences
    'max_normal_sequences': None,    # None = use all, or set limit
}


# ============================================================================
# LABELING STRATEGY (PHASE 2 - SEMI-SUPERVISED)
# ============================================================================

LABELING_CONFIG = {
    # Stratified sampling
    'total_sequences_to_label': 150,
    
    'sampling_strategy': {
        'extreme_anomaly': {
            'percentile_range': (95, 100),  # Top 5%
            'n_samples': 30
        },
        'high_anomaly': {
            'percentile_range': (85, 95),   # 85-95th percentile
            'n_samples': 40
        },
        'medium_anomaly': {
            'percentile_range': (70, 85),
            'n_samples': 30
        },
        'low_anomaly': {
            'percentile_range': (50, 70),
            'n_samples': 20
        },
        'normal_baseline': {
            'percentile_range': (0, 50),    # Bottom 50%
            'n_samples': 30
        }
    },
    
    # Label categories
    'fault_labels': {
        'binary': ['NO_FAULT', 'FAULT'],
        'detailed': [
            'thermal_overheat',
            'thermal_underheat',
            'sensor_drift',
            'sensor_noise',
            'sensor_disagreement',
            'vacuum_leak',
            'throttle_issue',
            'rpm_instability',
            'mechanical_fault',
            'electrical_fault',
            'false_alarm'
        ]
    },
    
    # Quality control
    'uncertainty_flag_threshold': 0.7,  # Flag if unsure
    'inter_rater_sample_size': 15,       # Double-label for consistency
}


# ============================================================================
# LSTM CLASSIFIER (PHASE 3 - SUPERVISED LEARNING)
# ============================================================================

CLASSIFIER_CONFIG = {
    # Architecture
    'input_shape': (1200, 10),
    'use_bidirectional': True,
    'lstm_units': [128, 64],
    'attention': True,
    'attention_type': 'global',     # 'global' or 'self'
    'dense_units': [32],
    'dropout': 0.3,
    'recurrent_dropout': 0.0,       # Set to 0 to enable cuDNN (3-5x faster!)
    'output_activation': 'sigmoid',
    
    # Training
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy', 'precision', 'recall', 'auc'],
    
    # Class imbalance handling
    'use_class_weights': True,
    'class_weight_method': 'computed',  # 'computed' or 'manual'
    'class_weight_adjustment': 0.8,     # Dampen computed weights
    'manual_class_weights': {
        0: 1.0,   # Normal
        1: 45.0   # Fault (adjust based on your data)
    },
    
    # Data augmentation (optional, for normal sequences only)
    'augmentation': {
        'enabled': False,
        'noise_std': 0.01,
        'scaling_range': (0.95, 1.05),
        'augmentation_factor': 2  # Generate 2x normal sequences
    },
    
    # Validation
    'validation_strategy': 'trip_based_split',  # Don't mix trips
    'test_size': 0.2,  # 20% of trips for testing
    'stratify': True,  # Maintain fault distribution
    
    # Callbacks
    'early_stopping': {
        'monitor': 'val_recall',
        'patience': 20,
        'mode': 'max',
        'restore_best_weights': True
    },
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 10,
        'min_lr': 0.00001
    },
    'model_checkpoint': {
        'monitor': 'val_f1',  # Custom F1 metric
        'save_best_only': True,
        'mode': 'max'
    }
}


# ============================================================================
# THRESHOLD TUNING (PHASE 5 - OPTIMIZED)
# ============================================================================

THRESHOLD_CONFIG = {
    # Phase 5 Optimization Results (from 05_threshold_tuning.py)
    'optimal_threshold': 0.520,      # Optimized threshold from Phase 5
    'use_optimal_threshold': True,   # Set to False to use default 0.50
    'default_threshold': 0.500,      # Original default
    
    # Performance at optimal threshold
    'expected_performance': {
        'recall': 0.905,             # 90.5% (same as default)
        'precision': 0.383,          # 38.3% (+0.8% improvement)
        'f2_score': 0.711,           # (+0.6% improvement)
        'false_alarms': 92,          # -3 compared to default
        'missed_faults': 6,          # Same as default
    },
    
    # Optimization targets (from Phase 5 search)
    'primary_metric': 'f2_score',    # F-beta with beta=2 (recall weighted 2x)
    'min_recall': 0.88,              # Minimum acceptable recall
    'target_recall': 0.90,           # Target recall (achieved!)
    'min_precision': 0.40,           # Minimum desired precision
    
    # Search parameters (for re-optimization)
    'search_range': (0.20, 0.80),    # Threshold search range
    'search_step': 0.01,             # Search step size
    'optimization_metric': 'f2_score',  # Metric to optimize
    'beta': 2.0,                     # F-beta weight (recall 2x precision)
    
    # Business costs (for cost-based optimization)
    'cost_false_alarm': 75,          # Cost of diagnostic check ($)
    'cost_missed_fault': 1000,       # Cost of breakdown ($)
    
    # Per-vehicle thresholds (future enhancement)
    'enable_vehicle_thresholds': False,  # Not yet implemented
    'min_trips_for_personalization': 50,
    
    # Safety bounds
    'absolute_min_threshold': 0.20,  # Never go below this
    'absolute_max_threshold': 0.70,  # Never go above this
    
    # Monitoring
    'monitor_threshold_performance': True,
    'alert_if_recall_drops_below': 0.88,
}


# ============================================================================
# ONLINE LEARNING (PHASE 4 - VEHICLE ADAPTATION)
# ============================================================================

ONLINE_LEARNING_CONFIG = {
    # Adaptation layer architecture
    'adaptation_layer_units': [16],
    'adaptation_dropout': 0.2,
    'adaptation_activation': 'relu',
    
    # Learning parameters
    'learning_rate': 0.0005,         # Lower than base model
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    
    # Update policy
    'update_trigger': 'high_confidence',
    'confidence_threshold': 0.80,     # Only learn from confident predictions
    'min_trips_before_adaptation': 10,  # Need baseline before adapting
    'update_frequency': 'per_trip',
    
    # Drift detection
    'enable_drift_detection': True,
    'drift_threshold': 0.3,          # Alert if offset > 0.3
    'drift_window': 20,              # Check over last 20 trips
    
    # Threshold adaptation
    'adaptive_threshold': True,
    'threshold_update_rate': 0.05,   # Gradual adjustment
    
    # Storage & persistence
    'save_vehicle_weights': True,
    'vehicle_db_path': '/mnt/user-data/outputs/vehicle_adaptations.json',
    'backup_frequency': 'daily',
}


# ============================================================================
# GPU CONFIGURATION
# ============================================================================

GPU_CONFIG = {
    # TensorFlow GPU settings
    'use_gpu': True,
    'gpu_memory_limit': 3500,        # MB (leave 500MB for system)
    'gpu_memory_growth': True,       # Allocate incrementally
    
    # Mixed precision (FP16)
    'mixed_precision': True,         # 2x faster on RTX 3050!
    
    # Memory optimization
    'clear_session_after_training': True,
    'gradient_accumulation_steps': 1,  # Set to 2 if OOM
    
    # Batch size fallback
    'auto_reduce_batch_on_oom': True,
    'min_batch_size': 8,
}


# ============================================================================
# MONITORING & LOGGING
# ============================================================================

MONITORING_CONFIG = {
    # TensorBoard
    'use_tensorboard': True,
    'tensorboard_log_dir': '/mnt/user-data/outputs/logs/tensorboard',
    
    # Model checkpoints
    'checkpoint_frequency': 'epoch',  # Save after each epoch
    'keep_n_checkpoints': 5,          # Keep best 5
    
    # Metrics to track
    'track_metrics': [
        'loss', 'val_loss',
        'accuracy', 'val_accuracy',
        'precision', 'val_precision',
        'recall', 'val_recall',
        'auc', 'val_auc',
        'f1', 'val_f1'
    ],
    
    # Alerts
    'enable_alerts': True,
    'alert_conditions': {
        'overfitting': {
            'metric': 'val_loss',
            'threshold': 'train_loss * 1.5',
            'message': 'Model may be overfitting!'
        },
        'poor_recall': {
            'metric': 'val_recall',
            'threshold': 0.80,
            'message': 'Recall dropped below 80%'
        }
    }
}


# ============================================================================
# EVALUATION METRICS
# ============================================================================

EVALUATION_CONFIG = {
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'roc_auc',
        'pr_auc',
        'confusion_matrix',
        'classification_report'
    ],
    
    # Visualization
    'plot_roc_curve': True,
    'plot_precision_recall_curve': True,
    'plot_confusion_matrix': True,
    'plot_attention_heatmap': True,
    
    # Per-class metrics
    'per_class_breakdown': True,
    
    # Confidence calibration
    'plot_calibration_curve': True,
}


# ============================================================================
# RANDOM SEEDS (Reproducibility)
# ============================================================================

RANDOM_SEEDS = {
    'numpy': 42,
    'tensorflow': 42,
    'python': 42
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_summary():
    """Print configuration summary"""
    print("=" * 80)
    print("LSTM VEHICLE DIAGNOSTICS - CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\nSequence Length: {DATA_CONFIG['sequence_length']} samples ({DATA_CONFIG['sequence_length']/DATA_CONFIG['sampling_rate']:.0f} seconds)")
    print(f"Features: {len(DATA_CONFIG['feature_columns'])}")
    print(f"Autoencoder Latent Dim: {AUTOENCODER_CONFIG['latent_dim']}")
    print(f"Classifier LSTM Units: {CLASSIFIER_CONFIG['lstm_units']}")
    print(f"Batch Size: {CLASSIFIER_CONFIG['batch_size']}")
    print(f"Target Recall: {THRESHOLD_CONFIG['min_recall']:.0%}")
    print(f"GPU Memory Limit: {GPU_CONFIG['gpu_memory_limit']}MB")
    print(f"Mixed Precision: {GPU_CONFIG['mixed_precision']}")
    print("=" * 80)


if __name__ == "__main__":
    get_config_summary()
