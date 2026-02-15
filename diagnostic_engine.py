"""
DIAGNOSTIC ENGINE - 3-TIER FAULT CLASSIFICATION
================================================

Maps LSTM predictions + attention → Specific fault categories

Tiers:
- Tier 1 (Precise): High confidence (>0.80), specific fault category
- Tier 2 (General): Medium confidence (0.60-0.80), possible categories
- Tier 3 (Failsafe): Low confidence (<0.60) OR critical violation

Fault Categories:
1. SENSOR_FAULT
2. AIR_SUPPLY_FAULT
3. IGNITION_COMBUSTION_FAULT
4. THROTTLE_CONTROL_FAULT
5. RPM_MECHANICAL_FAULT
6. THERMAL_MANAGEMENT_FAULT
7. ELECTRICAL_COMMUNICATION_FAULT
8. UNKNOWN_FAULT
"""

import numpy as np


# ============================================================================
# FAULT CATEGORY DEFINITIONS
# ============================================================================

FAULT_CATEGORIES = {
    'SENSOR_FAULT': {
        'subsystem': 'Sensor Network',
        'description': 'Sensor malfunction or measurement error',
        'examples': [
            'Coolant sensor reading impossible values',
            'IAT sensor drift',
            'MAF sensor failure',
            'APP sensor disagreement (D ≠ E)'
        ],
        'urgency_default': 'MEDIUM',
        'repair_time': '30-90 minutes',
        'estimated_cost': '$50-200'
    },
    
    'AIR_SUPPLY_FAULT': {
        'subsystem': 'Air Intake System',
        'description': 'Air supply or intake system malfunction',
        'examples': [
            'Vacuum leak',
            'MAP-throttle mismatch',
            'Air filter restriction',
            'Intake manifold issue'
        ],
        'urgency_default': 'HIGH',
        'repair_time': '1-2 hours',
        'estimated_cost': '$100-400'
    },
    
    'IGNITION_COMBUSTION_FAULT': {
        'subsystem': 'Engine Combustion',
        'description': 'Combustion or ignition system issue',
        'examples': [
            'Misfire detected',
            'Poor combustion efficiency',
            'Spark plug failure',
            'Ignition coil problem'
        ],
        'urgency_default': 'HIGH',
        'repair_time': '1-3 hours',
        'estimated_cost': '$150-600'
    },
    
    'THROTTLE_CONTROL_FAULT': {
        'subsystem': 'Throttle System',
        'description': 'Throttle control malfunction',
        'examples': [
            'Electronic throttle control fault',
            'APP sensor mismatch',
            'Throttle body issue',
            'Throttle position sensor drift'
        ],
        'urgency_default': 'HIGH',
        'repair_time': '1-2 hours',
        'estimated_cost': '$200-500'
    },
    
    'RPM_MECHANICAL_FAULT': {
        'subsystem': 'Engine Mechanical',
        'description': 'Mechanical engine issue affecting RPM',
        'examples': [
            'Idle control valve failure',
            'Timing belt/chain issue',
            'Engine mount problem',
            'Crankshaft sensor fault'
        ],
        'urgency_default': 'HIGH',
        'repair_time': '2-4 hours',
        'estimated_cost': '$200-800'
    },
    
    'THERMAL_MANAGEMENT_FAULT': {
        'subsystem': 'Cooling System',
        'description': 'Engine thermal management issue',
        'examples': [
            'Overheating condition',
            'Thermostat failure',
            'Coolant leak',
            'Radiator/water pump malfunction'
        ],
        'urgency_default': 'CRITICAL',
        'repair_time': '1-3 hours',
        'estimated_cost': '$150-600'
    },
    
    'ELECTRICAL_COMMUNICATION_FAULT': {
        'subsystem': 'ECU/Wiring',
        'description': 'Electrical or communication system fault',
        'examples': [
            'ECU communication error',
            'Wiring short or open circuit',
            'CAN bus issue',
            'Ground fault'
        ],
        'urgency_default': 'HIGH',
        'repair_time': '2-5 hours',
        'estimated_cost': '$200-1000'
    },
    
    'UNKNOWN_FAULT': {
        'subsystem': 'Unclassified',
        'description': 'Fault detected but pattern unclear',
        'examples': ['Anomaly detected but specific cause unknown'],
        'urgency_default': 'MEDIUM',
        'repair_time': 'Diagnostic required',
        'estimated_cost': 'TBD'
    }
}


# ============================================================================
# DIAGNOSTIC ENGINE
# ============================================================================

class DiagnosticEngine:
    """
    3-tier diagnostic system with fault categorization
    """
    
    def __init__(self):
        self.feature_names = [
            'coolant_temp', 'map', 'rpm', 'speed', 'intake_temp',
            'maf', 'throttle', 'ambient_temp', 'app_d', 'app_e'
        ]
    
    def diagnose(self, sequence, fault_probability, attention_weights=None):
        """
        Main diagnostic pipeline
        
        Args:
            sequence: (1200, 10) normalized sensor data
            fault_probability: float (0-1) from LSTM
            attention_weights: (1200, 1) attention scores
        
        Returns:
            dict: Diagnostic report
        """
        
        if fault_probability < 0.5:
            return {
                'tier': 0,
                'mode': 'NORMAL',
                'fault_detected': False,
                'confidence': 1 - fault_probability,
                'status': 'Vehicle operating normally'
            }
        
        # Extract features
        violations = self.check_physics_violations(sequence)
        
        if attention_weights is not None:
            critical_moments = self.extract_critical_timesteps(attention_weights)
            sensor_analysis = self.analyze_sensors_at_moments(sequence, critical_moments, attention_weights)
        else:
            critical_moments = []
            sensor_analysis = self.analyze_overall_sequence(sequence)
        
        # ═══════════════════════════════════════════════════════════════
        # TIER 1: PRECISE DIAGNOSIS (High Confidence)
        # ═══════════════════════════════════════════════════════════════
        if fault_probability > 0.80:
            fault_category = self.classify_fault_precise(violations, sensor_analysis)
            
            if fault_category != 'UNKNOWN_FAULT':
                return self.generate_tier1_report(
                    fault_category,
                    fault_probability,
                    violations,
                    sensor_analysis,
                    critical_moments
                )
        
        # ═══════════════════════════════════════════════════════════════
        # TIER 2: GENERAL WARNING (Medium Confidence)
        # ═══════════════════════════════════════════════════════════════
        if 0.60 <= fault_probability <= 0.80:
            possible_categories = self.classify_fault_general(violations, sensor_analysis)
            
            return self.generate_tier2_report(
                possible_categories,
                fault_probability,
                sensor_analysis
            )
        
        # ═══════════════════════════════════════════════════════════════
        # TIER 3: SAFETY FALLBACK (Low Confidence OR Critical)
        # ═══════════════════════════════════════════════════════════════
        critical_violations = [v for v in violations if violations[v].get('critical', False)]
        
        return self.generate_tier3_report(
            fault_probability,
            critical_violations if critical_violations else list(violations.keys()),
            sensor_analysis
        )
    
    def check_physics_violations(self, sequence):
        """
        Check for physics-based violations
        
        Returns dict of violations with evidence
        """
        violations = {}
        
        # Denormalize (approximate ranges for checking)
        coolant = sequence[:, 0] * 150  # 0-150°C
        MAP = sequence[:, 1] * 120      # 0-120 kPa
        RPM = sequence[:, 2] * 7000     # 0-7000 RPM
        speed = sequence[:, 3] * 250    # 0-250 km/h
        intake_temp = sequence[:, 4] * 150
        MAF = sequence[:, 5] * 300      # 0-300 g/s
        throttle = sequence[:, 6] * 100
        ambient = sequence[:, 7] * 70
        APP_D = sequence[:, 8] * 100
        APP_E = sequence[:, 9] * 100
        
        # ════════════════════════════════════════════════════════════
        # SENSOR_FAULT indicators
        # ════════════════════════════════════════════════════════════
        
        # IAT < Ambient (impossible)
        iat_violations = np.where(intake_temp < ambient - 5)[0]
        if len(iat_violations) > 50:  # More than 5 seconds
            violations['iat_below_ambient'] = {
                'type': 'SENSOR_FAULT',
                'evidence': f'IAT below ambient at {len(iat_violations)} points',
                'severity': 'MEDIUM',
                'critical': False
            }
        
        # Coolant overheat
        if np.max(coolant) > 120:
            violations['coolant_overheat'] = {
                'type': 'THERMAL_MANAGEMENT_FAULT',
                'evidence': f'Coolant temperature {np.max(coolant):.1f}°C (>120°C)',
                'severity': 'CRITICAL',
                'critical': True
            }
        
        # Impossible coolant jump
        coolant_jumps = np.abs(np.diff(coolant))
        max_jump = np.max(coolant_jumps) if len(coolant_jumps) > 0 else 0
        if max_jump > 20:  # 20°C in 0.1s
            violations['coolant_sensor_jump'] = {
                'type': 'SENSOR_FAULT',
                'evidence': f'Coolant jumped {max_jump:.1f}°C in 0.1s',
                'severity': 'HIGH',
                'critical': False
            }
        
        # APP sensor disagreement
        app_diff = np.abs(APP_D - APP_E)
        if np.mean(app_diff) > 5:
            violations['app_disagreement'] = {
                'type': 'THROTTLE_CONTROL_FAULT',
                'evidence': f'APP sensors disagree (avg {np.mean(app_diff):.1f}%)',
                'severity': 'HIGH',
                'critical': False
            }
        
        # ════════════════════════════════════════════════════════════
        # RPM_MECHANICAL_FAULT indicators
        # ════════════════════════════════════════════════════════════
        
        # RPM instability at idle
        idle_mask = speed < 5  # Car not moving
        if idle_mask.sum() > 100:  # At least 10 seconds
            idle_rpm = RPM[idle_mask]
            rpm_std = np.std(idle_rpm)
            if rpm_std > 200:
                violations['rpm_instability_at_idle'] = {
                    'type': 'RPM_MECHANICAL_FAULT',
                    'evidence': f'RPM unstable at idle (std={rpm_std:.0f})',
                    'severity': 'HIGH',
                    'critical': False
                }
        
        # ════════════════════════════════════════════════════════════
        # AIR_SUPPLY_FAULT indicators
        # ════════════════════════════════════════════════════════════
        
        # MAP-Throttle mismatch (WOT but low MAP)
        wot_mask = throttle > 90
        if wot_mask.sum() > 50:  # At least 5 seconds
            wot_map = MAP[wot_mask]
            if np.mean(wot_map) < 80:  # Should be near atmospheric
                violations['vacuum_leak'] = {
                    'type': 'AIR_SUPPLY_FAULT',
                    'evidence': f'WOT but low MAP ({np.mean(wot_map):.0f} kPa)',
                    'severity': 'HIGH',
                    'critical': False
                }
        
        # ════════════════════════════════════════════════════════════
        # ELECTRICAL_COMMUNICATION_FAULT indicators
        # ════════════════════════════════════════════════════════════
        
        # Multiple sensors stuck
        stuck_count = 0
        for i, sensor in enumerate(self.feature_names):
            sensor_data = sequence[:, i]
            # Check if sensor stuck (same value for >100 samples)
            unique_values = len(np.unique(sensor_data))
            if unique_values < 5:  # Very few unique values
                stuck_count += 1
        
        if stuck_count >= 3:
            violations['multiple_sensor_dropout'] = {
                'type': 'ELECTRICAL_COMMUNICATION_FAULT',
                'evidence': f'{stuck_count} sensors showing minimal variation',
                'severity': 'HIGH',
                'critical': False
            }
        
        return violations
    
    def extract_critical_timesteps(self, attention_weights, threshold_multiplier=2.0):
        """Extract timesteps with high attention"""
        attention_flat = attention_weights.flatten()
        mean_attention = np.mean(attention_flat)
        threshold = mean_attention * threshold_multiplier
        
        critical_indices = np.where(attention_flat > threshold)[0]
        return critical_indices.tolist()
    
    def analyze_sensors_at_moments(self, sequence, critical_moments, attention_weights):
        """Analyze which sensors are involved at critical moments"""
        
        if len(critical_moments) == 0:
            return self.analyze_overall_sequence(sequence)
        
        # Get sensor values at critical moments
        critical_data = sequence[critical_moments, :]
        
        # Find sensors with highest variation at critical moments
        sensor_variation = np.std(critical_data, axis=0)
        sensor_importance = sensor_variation / (np.std(sequence, axis=0) + 1e-8)
        
        # Get top sensors
        top_sensor_indices = np.argsort(sensor_importance)[::-1][:3]
        primary_sensors = [self.feature_names[i] for i in top_sensor_indices]
        
        # Analyze patterns
        patterns = {}
        
        # Check for jumps
        for idx in top_sensor_indices:
            sensor_data = sequence[:, idx]
            jumps = np.abs(np.diff(sensor_data))
            if np.max(jumps) > 0.3:  # Normalized jump >0.3
                patterns['jump_detected'] = True
                patterns['jump_sensor'] = self.feature_names[idx]
                patterns['jump_magnitude'] = float(np.max(jumps))
                break
        
        # Check for oscillations
        for idx in top_sensor_indices:
            sensor_data = sequence[:, idx]
            if np.std(sensor_data) > 0.2:  # High variation
                patterns['oscillation_detected'] = True
                patterns['oscillation_sensor'] = self.feature_names[idx]
                break
        
        return {
            'primary_sensors': primary_sensors,
            'critical_moments': critical_moments,
            'patterns': patterns,
            'sensor_importance': {self.feature_names[i]: float(sensor_importance[i]) 
                                 for i in top_sensor_indices}
        }
    
    def analyze_overall_sequence(self, sequence):
        """Analyze entire sequence when no attention available"""
        # Find sensors with highest overall variation
        sensor_variation = np.std(sequence, axis=0)
        top_sensor_indices = np.argsort(sensor_variation)[::-1][:3]
        primary_sensors = [self.feature_names[i] for i in top_sensor_indices]
        
        return {
            'primary_sensors': primary_sensors,
            'critical_moments': [],
            'patterns': {},
            'sensor_importance': {}
        }
    
    def classify_fault_precise(self, violations, sensor_analysis):
        """
        Tier 1: Precise fault classification
        Returns specific fault category
        """
        primary_sensors = sensor_analysis.get('primary_sensors', [])
        patterns = sensor_analysis.get('patterns', {})
        
        # Priority 1: Direct violation mapping
        for violation, info in violations.items():
            if info.get('critical', False):
                return info['type']
        
        # Priority 2: Clear violation types
        for violation, info in violations.items():
            violation_type = info['type']
            if violation_type != 'UNKNOWN_FAULT':
                return violation_type
        
        # Priority 3: Sensor-based inference
        if 'coolant_temp' in primary_sensors:
            if patterns.get('jump_detected'):
                return 'SENSOR_FAULT'
            return 'THERMAL_MANAGEMENT_FAULT'
        
        if 'rpm' in primary_sensors:
            if patterns.get('oscillation_detected'):
                return 'RPM_MECHANICAL_FAULT'
            return 'IGNITION_COMBUSTION_FAULT'
        
        if 'map' in primary_sensors or 'maf' in primary_sensors:
            return 'AIR_SUPPLY_FAULT'
        
        if 'throttle' in primary_sensors or 'app_d' in primary_sensors or 'app_e' in primary_sensors:
            return 'THROTTLE_CONTROL_FAULT'
        
        if len(primary_sensors) >= 4:
            return 'ELECTRICAL_COMMUNICATION_FAULT'
        
        return 'UNKNOWN_FAULT'
    
    def classify_fault_general(self, violations, sensor_analysis):
        """
        Tier 2: General fault classification
        Returns list of 2-3 possible categories
        """
        possible = set()
        
        # Add all violation types
        for violation, info in violations.items():
            possible.add(info['type'])
        
        # Add sensor-based possibilities
        primary_sensors = sensor_analysis.get('primary_sensors', [])
        
        if 'coolant_temp' in primary_sensors:
            possible.add('THERMAL_MANAGEMENT_FAULT')
            possible.add('SENSOR_FAULT')
        
        if 'rpm' in primary_sensors:
            possible.add('RPM_MECHANICAL_FAULT')
            possible.add('IGNITION_COMBUSTION_FAULT')
        
        if 'map' in primary_sensors or 'maf' in primary_sensors:
            possible.add('AIR_SUPPLY_FAULT')
        
        # Return top 2-3
        possible.discard('UNKNOWN_FAULT')
        return list(possible)[:3] if len(possible) > 0 else ['UNKNOWN_FAULT']
    
    def generate_tier1_report(self, fault_category, confidence, violations, sensor_analysis, critical_moments):
        """Generate Tier 1 (Precise) diagnostic report"""
        category_info = FAULT_CATEGORIES[fault_category]
        
        # Build evidence
        evidence = {}
        for violation, info in violations.items():
            if info['type'] == fault_category:
                evidence[violation] = info['evidence']
        
        # Add sensor evidence
        if critical_moments:
            evidence['critical_timesteps'] = f"{len(critical_moments)} high-attention moments"
            if len(critical_moments) > 0:
                time_range = f"{min(critical_moments)/10:.1f}-{max(critical_moments)/10:.1f}s"
                evidence['time_window'] = time_range
        
        evidence['primary_sensors'] = ', '.join(sensor_analysis.get('primary_sensors', []))
        
        # Generate recommendation
        recommendation = self.get_recommendation(fault_category, violations)
        urgency = self.assess_urgency(fault_category, violations)
        
        return {
            'tier': 1,
            'mode': 'PRECISE',
            'fault_detected': True,
            'confidence': float(confidence),
            'category': fault_category,
            'subsystem': category_info['subsystem'],
            'description': category_info['description'],
            'evidence': evidence,
            'affected_sensors': sensor_analysis.get('primary_sensors', []),
            'recommendation': recommendation,
            'urgency': urgency,
            'estimated_repair_time': category_info['repair_time'],
            'estimated_cost': category_info['estimated_cost']
        }
    
    def generate_tier2_report(self, possible_categories, confidence, sensor_analysis):
        """Generate Tier 2 (General) diagnostic report"""
        return {
            'tier': 2,
            'mode': 'GENERAL',
            'fault_detected': True,
            'confidence': float(confidence),
            'possible_categories': possible_categories,
            'subsystems': [FAULT_CATEGORIES[cat]['subsystem'] for cat in possible_categories],
            'primary_sensors': sensor_analysis.get('primary_sensors', []),
            'recommendation': 
                'Fault detected but specific cause unclear. '
                'Perform comprehensive diagnostic scan. '
                f'Focus on: {", ".join(sensor_analysis.get("primary_sensors", []))}',
            'urgency': 'MEDIUM'
        }
    
    def generate_tier3_report(self, confidence, violations, sensor_analysis):
        """Generate Tier 3 (Safety Fallback) diagnostic report"""
        # Infer affected subsystems from sensors
        primary_sensors = sensor_analysis.get('primary_sensors', [])
        subsystems = set()
        
        for sensor in primary_sensors:
            if sensor in ['coolant_temp']:
                subsystems.add('Cooling System')
            elif sensor in ['rpm']:
                subsystems.add('Engine')
            elif sensor in ['map', 'maf', 'intake_temp']:
                subsystems.add('Air Intake')
            elif sensor in ['throttle', 'app_d', 'app_e']:
                subsystems.add('Throttle Control')
        
        return {
            'tier': 3,
            'mode': 'SAFETY_FALLBACK',
            'fault_detected': True,
            'confidence': float(confidence),
            'warning': 'ANOMALY DETECTED - PATTERN UNCLEAR',
            'violations': violations if violations else ['Model confidence low'],
            'affected_subsystems': list(subsystems) if subsystems else ['Engine'],
            'primary_sensors': primary_sensors,
            'recommendation': 
                'IMMEDIATE INSPECTION REQUIRED. '
                'Unusual vehicle behavior detected but specific cause not identified. '
                'Perform full diagnostic scan.',
            'urgency': 'HIGH',
            'safety_note': 'Low confidence diagnosis. Do not ignore - vehicle requires inspection.'
        }
    
    def get_recommendation(self, fault_category, violations):
        """Get specific recommendation based on fault category"""
        recommendations = {
            'SENSOR_FAULT': 
                'Inspect sensor wiring and connections. Check for sensor drift or failure. '
                'Verify ECU communication. Sensor replacement may be required.',
            
            'AIR_SUPPLY_FAULT':
                'Check for vacuum leaks in intake manifold and hoses. '
                'Inspect throttle body and air filter. Verify MAP sensor operation.',
            
            'IGNITION_COMBUSTION_FAULT':
                'Check spark plugs and ignition coils. Inspect fuel injectors. '
                'Verify compression. May require misfire diagnostic.',
            
            'THROTTLE_CONTROL_FAULT':
                'Inspect electronic throttle control system. Verify APP sensor calibration. '
                'Check throttle body for sticking or carbon buildup.',
            
            'RPM_MECHANICAL_FAULT':
                'Inspect idle air control valve. Check timing belt/chain. '
                'Verify crankshaft position sensor. Check for vacuum leaks.',
            
            'THERMAL_MANAGEMENT_FAULT':
                'CRITICAL: Check coolant level immediately. Inspect for leaks. '
                'Verify thermostat operation. Check radiator and water pump. '
                'DO NOT CONTINUE DRIVING IF OVERHEATING.',
            
            'ELECTRICAL_COMMUNICATION_FAULT':
                'Check ECU connections and ground points. Inspect wiring harness. '
                'Verify battery voltage. May require ECU diagnostic scan.',
            
            'UNKNOWN_FAULT':
                'Perform comprehensive diagnostic scan. Review all sensor data. '
                'Consult mechanic for detailed inspection.'
        }
        
        # Add specific violation details
        base_rec = recommendations.get(fault_category, '')
        
        if 'coolant_overheat' in violations:
            base_rec = 'CRITICAL: Engine overheating! Stop driving immediately. ' + base_rec
        
        return base_rec
    
    def assess_urgency(self, fault_category, violations):
        """Assess urgency level"""
        # Critical violations override
        for violation, info in violations.items():
            if info.get('critical', False):
                return 'CRITICAL'
        
        # Check for high severity
        high_severity_count = sum(1 for v in violations.values() if v.get('severity') == 'HIGH')
        if high_severity_count >= 2:
            return 'HIGH'
        
        # Default by category
        return FAULT_CATEGORIES[fault_category]['urgency_default']
