#!/usr/bin/env python3
"""
Resonance Signature Capture System
Advanced quantum resonance detection and signature analysis platform
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import json
import time
import threading
from datetime import datetime
import os
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)

class ResonanceCapture:
    def __init__(self):
        self.active_captures = {}
        self.signature_database = {}
        self.monitoring = False
        self.capture_thread = None
        
    def start_monitoring(self):
        """Start continuous resonance monitoring"""
        self.monitoring = True
        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.capture_thread = threading.Thread(target=self._monitor_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
    
    def stop_monitoring(self):
        """Stop resonance monitoring"""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Main monitoring loop for resonance detection"""
        while self.monitoring:
            try:
                # Simulate quantum resonance detection
                resonance_data = self._detect_resonance()
                if resonance_data['strength'] > 0.7:  # Threshold for significant resonance
                    signature = self._extract_signature(resonance_data)
                    self._store_signature(signature)
                time.sleep(0.1)  # 10Hz sampling rate
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def _detect_resonance(self):
        """Simulate advanced resonance detection"""
        # Generate synthetic quantum resonance patterns
        t = np.linspace(0, 1, 1000)
        
        # Base frequency with quantum fluctuations
        base_freq = 432.0 + np.random.normal(0, 0.1)  # Hz
        
        # Multi-harmonic resonance pattern
        signal_data = (
            np.sin(2 * np.pi * base_freq * t) +
            0.5 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 3 * t) +
            0.1 * np.random.normal(0, 1, len(t))  # Quantum noise
        )
        
        # Calculate resonance strength
        fft_data = fft(signal_data)
        freqs = fftfreq(len(t), t[1] - t[0])
        
        # Find peak resonance
        peak_idx = np.argmax(np.abs(fft_data[:len(fft_data)//2]))
        peak_freq = abs(freqs[peak_idx])
        peak_amplitude = abs(fft_data[peak_idx])
        
        # Normalize strength (0-1 scale)
        strength = min(peak_amplitude / 500.0, 1.0)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'frequency': peak_freq,
            'strength': strength,
            'raw_signal': signal_data.tolist(),
            'fft_data': np.abs(fft_data).tolist(),
            'frequencies': freqs.tolist()
        }
    
    def _extract_signature(self, resonance_data):
        """Extract unique signature from resonance data"""
        signature_id = f"sig_{int(time.time() * 1000)}"
        
        # Calculate signature characteristics
        signal_array = np.array(resonance_data['raw_signal'])
        
        signature = {
            'id': signature_id,
            'timestamp': resonance_data['timestamp'],
            'primary_frequency': resonance_data['frequency'],
            'strength': resonance_data['strength'],
            'harmonics': self._analyze_harmonics(resonance_data),
            'phase_pattern': self._extract_phase_pattern(signal_array),
            'coherence_factor': self._calculate_coherence(signal_array),
            'quantum_signature': self._generate_quantum_signature(signal_array)
        }
        
        return signature
    
    def _analyze_harmonics(self, resonance_data):
        """Analyze harmonic content of the resonance"""
        fft_data = np.array(resonance_data['fft_data'])
        freqs = np.array(resonance_data['frequencies'])
        
        # Find significant harmonics
        peaks, _ = signal.find_peaks(fft_data[:len(fft_data)//2], height=np.max(fft_data) * 0.1)
        
        harmonics = []
        for peak in peaks[:10]:  # Top 10 harmonics
            harmonics.append({
                'frequency': abs(freqs[peak]),
                'amplitude': fft_data[peak],
                'phase': np.angle(fft_data[peak])
            })
        
        return harmonics
    
    def _extract_phase_pattern(self, signal_array):
        """Extract phase pattern characteristics"""
        analytic_signal = signal.hilbert(signal_array)
        phase = np.unwrap(np.angle(analytic_signal))
        
        return {
            'mean_phase': np.mean(phase),
            'phase_variance': np.var(phase),
            'phase_drift': np.polyfit(range(len(phase)), phase, 1)[0]
        }
    
    def _calculate_coherence(self, signal_array):
        """Calculate quantum coherence factor"""
        # Simplified coherence calculation
        envelope = np.abs(signal.hilbert(signal_array))
        coherence = 1.0 - (np.std(envelope) / np.mean(envelope))
        return max(0, min(1, coherence))
    
    def _generate_quantum_signature(self, signal_array):
        """Generate unique quantum signature hash"""
        # Create a unique fingerprint based on signal characteristics
        features = [
            np.mean(signal_array),
            np.std(signal_array),
            np.max(signal_array),
            np.min(signal_array),
            len(signal.find_peaks(signal_array)[0])
        ]
        
        # Simple hash generation
        signature_hash = hash(tuple(np.round(features, 6))) % (10**12)
        return f"QS{signature_hash:012d}"
    
    def _store_signature(self, signature):
        """Store signature in database"""
        self.signature_database[signature['id']] = signature
        print(f"Captured signature: {signature['id']} - Strength: {signature['strength']:.3f}")
    
    def get_recent_signatures(self, limit=10):
        """Get most recent signatures"""
        signatures = list(self.signature_database.values())
        signatures.sort(key=lambda x: x['timestamp'], reverse=True)
        return signatures[:limit]
    
    def generate_visualization(self, signature_id):
        """Generate visualization for a specific signature"""
        if signature_id not in self.signature_database:
            return None
        
        signature = self.signature_database[signature_id]
        
        # Create plotly visualization
        fig = go.Figure()
        
        # Add harmonic data
        if signature['harmonics']:
            freqs = [h['frequency'] for h in signature['harmonics']]
            amps = [h['amplitude'] for h in signature['harmonics']]
            
            fig.add_trace(go.Scatter(
                x=freqs,
                y=amps,
                mode='markers+lines',
                name='Harmonic Spectrum',
                marker=dict(size=8, color='blue')
            ))
        
        fig.update_layout(
            title=f'Resonance Signature: {signature_id}',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Amplitude',
            template='plotly_dark'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Global resonance capture instance
resonance_system = ResonanceCapture()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    resonance_system.start_monitoring()
    return jsonify({'status': 'monitoring_started', 'timestamp': datetime.now().isoformat()})

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    resonance_system.stop_monitoring()
    return jsonify({'status': 'monitoring_stopped', 'timestamp': datetime.now().isoformat()})

@app.route('/api/status')
def get_status():
    return jsonify({
        'monitoring': resonance_system.monitoring,
        'total_signatures': len(resonance_system.signature_database),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/signatures')
def get_signatures():
    signatures = resonance_system.get_recent_signatures()
    return jsonify(signatures)

@app.route('/api/signature/<signature_id>')
def get_signature(signature_id):
    if signature_id in resonance_system.signature_database:
        return jsonify(resonance_system.signature_database[signature_id])
    return jsonify({'error': 'Signature not found'}), 404

@app.route('/api/visualization/<signature_id>')
def get_visualization(signature_id):
    viz_data = resonance_system.generate_visualization(signature_id)
    if viz_data:
        return jsonify({'visualization': viz_data})
    return jsonify({'error': 'Visualization not available'}), 404

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("🌊 Resonance Signature Capture System Starting...")
    print("🔬 Quantum resonance detection algorithms loaded")
    print("📡 Monitoring interface ready")
    print("🚀 System operational at http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)