"""
PHASE 4: SUPERVISED LSTM CLASSIFIER WITH DIAGNOSTIC ENGINE
===========================================================

Trains LSTM on clean labels to detect vehicle faults (not driving patterns).

Features:
- Bidirectional LSTM with Attention mechanism
- 3-tier diagnostic system (Precise/General/Failsafe)
- 8 fault categories (vehicle subsystems)
- Trip-based train/test split (no data leakage)
- Class weights for imbalance handling
- Comprehensive evaluation metrics
- Attention visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import warnings
warnings.filterwarnings('ignore')

from config import CLASSIFIER_CONFIG, PATHS, GPU_CONFIG, RANDOM_SEEDS

# Import diagnostic engine (we'll create this next)
from diagnostic_engine import DiagnosticEngine

# Set random seeds
np.random.seed(RANDOM_SEEDS['numpy'])
tf.random.set_seed(RANDOM_SEEDS['tensorflow'])


# ============================================================================
# GPU CONFIGURATION
# ============================================================================

if GPU_CONFIG['use_gpu']:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if GPU_CONFIG['gpu_memory_growth']:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=GPU_CONFIG['gpu_memory_limit']
                )]
            )
            print(f"✓ GPU configured: {gpus[0].name}")
            print(f"  Memory limit: {GPU_CONFIG['gpu_memory_limit']}MB")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    if GPU_CONFIG['mixed_precision']:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision enabled (FP16)")


# ============================================================================
# ATTENTION LAYER
# ============================================================================

class AttentionLayer(layers.Layer):
    """
    Global attention mechanism
    Learns which timesteps are important for fault detection
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        
        # Compute attention scores
        # (batch, timesteps, features) @ (features, 1) = (batch, timesteps, 1)
        e = keras.backend.tanh(keras.backend.dot(inputs, self.W) + self.b)
        
        # Attention weights (softmax over timesteps)
        a = keras.backend.softmax(e, axis=1)
        
        # Weighted sum: (batch, timesteps, features) * (batch, timesteps, 1)
        output = inputs * a
        output = keras.backend.sum(output, axis=1)
        
        return output, a
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# ============================================================================
# LSTM CLASSIFIER
# ============================================================================

class FaultClassifier:
    """
    LSTM-based fault classifier with attention and diagnostics
    """
    
    def __init__(self, config=CLASSIFIER_CONFIG):
        self.config = config
        self.model = None
        self.history = None
        self.attention_layer = None
        self.diagnostic_engine = DiagnosticEngine()
        
    def build_model(self, input_shape):
        """Build BiLSTM + Attention architecture"""
        
        print("\n" + "=" * 80)
        print("BUILDING LSTM CLASSIFIER")
        print("=" * 80)
        
        # Input
        inputs = layers.Input(shape=input_shape, name='sequence_input')
        
        # ====================================================================
        # BIDIRECTIONAL LSTM (reads forward and backward)
        # ====================================================================
        if self.config['use_bidirectional']:
            x = layers.Bidirectional(
                layers.LSTM(
                    self.config['lstm_units'][0],
                    return_sequences=True,
                    activation='tanh',
                    dropout=self.config['dropout'],
                    recurrent_dropout=self.config['recurrent_dropout'],
                    kernel_regularizer=keras.regularizers.l2(0.0001),
                    name='bilstm_1'
                ),
                name='bidirectional_lstm'
            )(inputs)
        else:
            x = layers.LSTM(
                self.config['lstm_units'][0],
                return_sequences=True,
                activation='tanh',
                dropout=self.config['dropout'],
                recurrent_dropout=self.config['recurrent_dropout'],
                name='lstm_1'
            )(inputs)
        
        # ====================================================================
        # ATTENTION LAYER
        # ====================================================================
        if self.config['attention']:
            attention_layer = AttentionLayer(name='attention')
            x, attention_weights = attention_layer(x)
            self.attention_layer = attention_layer
        else:
            # If no attention, just take last timestep
            x = x[:, -1, :]
        
        # ====================================================================
        # ADDITIONAL LSTM LAYER
        # ====================================================================
        if len(self.config['lstm_units']) > 1:
            x = layers.Reshape((1, -1))(x)  # Add time dimension back
            x = layers.LSTM(
                self.config['lstm_units'][1],
                activation='tanh',
                dropout=self.config['dropout'],
                recurrent_dropout=self.config['recurrent_dropout'],
                name='lstm_2'
            )(x)
        
        # ====================================================================
        # DENSE LAYERS
        # ====================================================================
        x = layers.Dropout(self.config['dropout'], name='dropout_1')(x)
        
        for i, units in enumerate(self.config['dense_units']):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.0001),
                name=f'dense_{i+1}'
            )(x)
        
        x = layers.Dropout(self.config['dropout'], name='dropout_2')(x)
        
        # ====================================================================
        # OUTPUT LAYER
        # ====================================================================
        outputs = layers.Dense(
            1,
            activation=self.config['output_activation'],
            name='fault_probability'
        )(x)
        
        # ====================================================================
        # BUILD MODEL
        # ====================================================================
        self.model = Model(inputs=inputs, outputs=outputs, name='fault_classifier')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=self.config['loss'],
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print(f"\nModel architecture:")
        self.model.summary()
        
        return self.model
    
    def prepare_data(self, sequences, labels, metadata):
        """
        Prepare data with trip-based split
        
        Critical: Split by TRIP, not by sequence (prevents data leakage)
        """
        print("\n" + "=" * 80)
        print("PREPARING DATA")
        print("=" * 80)
        
        # Get unique trips
        unique_trips = metadata['trip_id'].unique()
        n_trips = len(unique_trips)
        
        print(f"\nTotal trips: {n_trips}")
        print(f"Total sequences: {len(sequences)}")
        print(f"Label distribution:")
        print(f"  Normal (0): {(labels==0).sum()} ({(labels==0).sum()/len(labels)*100:.1f}%)")
        print(f"  Fault (1):  {(labels==1).sum()} ({(labels==1).sum()/len(labels)*100:.1f}%)")
        
        # Calculate fault rate per trip
        trip_fault_rates = metadata.groupby('trip_id')['label'].mean()
        
        # Stratified trip split (maintain fault distribution)
        train_trips, test_trips = train_test_split(
            unique_trips,
            test_size=self.config['test_size'],
            random_state=RANDOM_SEEDS['numpy'],
            stratify=(trip_fault_rates > 0).astype(int)  # Stratify by "has faults"
        )
        
        # Get sequence indices for each split
        train_mask = metadata['trip_id'].isin(train_trips)
        test_mask = metadata['trip_id'].isin(test_trips)
        
        X_train = sequences[train_mask]
        y_train = labels[train_mask]
        X_test = sequences[test_mask]
        y_test = labels[test_mask]
        
        train_metadata = metadata[train_mask].copy()
        test_metadata = metadata[test_mask].copy()
        
        print(f"\nTrain/Test Split:")
        print(f"  Train trips: {len(train_trips)} ({len(train_trips)/n_trips*100:.1f}%)")
        print(f"  Test trips:  {len(test_trips)} ({len(test_trips)/n_trips*100:.1f}%)")
        print(f"\n  Train sequences: {len(X_train)} ({(y_train==0).sum()} normal, {(y_train==1).sum()} fault)")
        print(f"  Test sequences:  {len(X_test)} ({(y_test==0).sum()} normal, {(y_test==1).sum()} fault)")
        
        # Compute class weights
        if self.config['use_class_weights']:
            if self.config['class_weight_method'] == 'computed':
                n_normal = (y_train == 0).sum()
                n_fault = (y_train == 1).sum()
                
                # Compute balanced class weights
                total = n_normal + n_fault
                weight_normal = 1.0
                weight_fault = (n_normal / n_fault) * self.config['class_weight_adjustment']
                
                self.class_weight = {0: weight_normal, 1: weight_fault}
            else:
                self.class_weight = self.config['manual_class_weights']
            
            print(f"\nClass weights:")
            print(f"  Normal (0): {self.class_weight[0]:.2f}")
            print(f"  Fault (1):  {self.class_weight[1]:.2f}")
        else:
            self.class_weight = None
        
        return (X_train, y_train, train_metadata), (X_test, y_test, test_metadata)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the classifier"""
        
        print("\n" + "=" * 80)
        print("TRAINING CLASSIFIER")
        print("=" * 80)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor=self.config['early_stopping']['monitor'],
                patience=self.config['early_stopping']['patience'],
                mode=self.config['early_stopping']['mode'],
                restore_best_weights=self.config['early_stopping']['restore_best_weights'],
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=self.config['reduce_lr']['monitor'],
                factor=self.config['reduce_lr']['factor'],
                patience=self.config['reduce_lr']['patience'],
                min_lr=self.config['reduce_lr']['min_lr'],
                verbose=1
            ),
            ModelCheckpoint(
                filepath=PATHS['classifier_model'],
                monitor=self.config['model_checkpoint']['monitor'],
                save_best_only=self.config['model_checkpoint']['save_best_only'],
                mode=self.config['model_checkpoint']['mode'],
                verbose=1
            )
        ]
        
        print(f"\nStarting training...")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        
        # Use validation split or separate validation set
        if X_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
            validation_split = 0.15
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=validation_data,
            validation_split=validation_split,
            class_weight=self.class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
        return self.history
    
    def evaluate(self, X_test, y_test, test_metadata):
        """Comprehensive evaluation"""
        
        print("\n" + "=" * 80)
        print("EVALUATING MODEL")
        print("=" * 80)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, batch_size=self.config['batch_size'], verbose=1)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()
        
        # Metrics
        accuracy = np.mean(y_pred == y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n{'='*80}")
        print("PERFORMANCE METRICS")
        print(f"{'='*80}")
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f} {'✓' if recall >= 0.90 else '⚠️ (target: 0.90)'}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Normal  Fault")
        print(f"Actual Normal   {cm[0,0]:4d}   {cm[0,1]:4d}  ← False alarms")
        print(f"       Fault    {cm[1,0]:4d}   {cm[1,1]:4d}  ← Missed faults")
        
        # Save predictions
        predictions_df = test_metadata.copy()
        predictions_df['y_true'] = y_test
        predictions_df['y_pred'] = y_pred
        predictions_df['y_pred_proba'] = y_pred_proba
        predictions_df.to_csv(f"{PATHS['output_dir']}/predictions.csv", index=False)
        print(f"\n✓ Predictions saved: {PATHS['output_dir']}/predictions.csv")
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'test_size': len(y_test),
            'threshold': 0.5
        }
        
        with open(f"{PATHS['output_dir']}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics, predictions_df
    
    def extract_attention_weights(self, sequence):
        """
        Extract attention weights for a single sequence
        """
        if not self.config['attention']:
            return None
        
        # Create intermediate model that outputs attention
        attention_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('attention').output[1]
        )
        
        # Get attention weights
        attention_weights = attention_model.predict(np.expand_dims(sequence, 0), verbose=0)
        
        return attention_weights.squeeze()


# Continued in next file...
