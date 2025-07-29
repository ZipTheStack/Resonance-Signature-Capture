#!/usr/bin/env python3
"""
Test suite for Resonance Signature Capture System
"""

import pytest
import numpy as np
import time
import threading
from app import ResonanceCapture

class TestResonanceCapture:
    
    def setup_method(self):
        """Setup test environment"""
        self.resonance_system = ResonanceCapture()
    
    def test_initialization(self):
        """Test system initialization"""
        assert self.resonance_system.active_captures == {}
        assert self.resonance_system.signature_database == {}
        assert self.resonance_system.monitoring == False
        assert self.resonance_system.capture_thread is None
    
    def test_resonance_detection(self):
        """Test resonance detection functionality"""
        resonance_data = self.resonance_system._detect_resonance()
        
        # Verify data structure
        assert 'timestamp' in resonance_data
        assert 'frequency' in resonance_data
        assert 'strength' in resonance_data
        assert 'raw_signal' in resonance_data
        assert 'fft_data' in resonance_data
        assert 'frequencies' in resonance_data
        
        # Verify data types and ranges
        assert isinstance(resonance_data['frequency'], (int, float))
        assert 0 <= resonance_data['strength'] <= 1
        assert len(resonance_data['raw_signal']) == 1000
        assert resonance_data['frequency'] > 0
    
    def test_signature_extraction(self):
        """Test signature extraction from resonance data"""
        resonance_data = self.resonance_system._detect_resonance()
        signature = self.resonance_system._extract_signature(resonance_data)
        
        # Verify signature structure
        assert 'id' in signature
        assert 'timestamp' in signature
        assert 'primary_frequency' in signature
        assert 'strength' in signature
        assert 'harmonics' in signature
        assert 'phase_pattern' in signature
        assert 'coherence_factor' in signature
        assert 'quantum_signature' in signature
        
        # Verify signature data
        assert signature['id'].startswith('sig_')
        assert isinstance(signature['harmonics'], list)
        assert 0 <= signature['coherence_factor'] <= 1
        assert signature['quantum_signature'].startswith('QS')
    
    def test_harmonic_analysis(self):
        """Test harmonic analysis functionality"""
        resonance_data = self.resonance_system._detect_resonance()
        harmonics = self.resonance_system._analyze_harmonics(resonance_data)
        
        assert isinstance(harmonics, list)
        for harmonic in harmonics:
            assert 'frequency' in harmonic
            assert 'amplitude' in harmonic
            assert 'phase' in harmonic
            assert harmonic['frequency'] >= 0
    
    def test_phase_pattern_extraction(self):
        """Test phase pattern extraction"""
        signal_array = np.random.randn(1000)
        phase_pattern = self.resonance_system._extract_phase_pattern(signal_array)
        
        assert 'mean_phase' in phase_pattern
        assert 'phase_variance' in phase_pattern
        assert 'phase_drift' in phase_pattern
        assert isinstance(phase_pattern['mean_phase'], (int, float))
        assert phase_pattern['phase_variance'] >= 0
    
    def test_coherence_calculation(self):
        """Test quantum coherence calculation"""
        signal_array = np.random.randn(1000)
        coherence = self.resonance_system._calculate_coherence(signal_array)
        
        assert isinstance(coherence, (int, float))
        assert 0 <= coherence <= 1
    
    def test_quantum_signature_generation(self):
        """Test quantum signature generation"""
        signal_array = np.random.randn(1000)
        quantum_sig = self.resonance_system._generate_quantum_signature(signal_array)
        
        assert isinstance(quantum_sig, str)
        assert quantum_sig.startswith('QS')
        assert len(quantum_sig) == 14  # QS + 12 digits
    
    def test_signature_storage(self):
        """Test signature storage functionality"""
        resonance_data = self.resonance_system._detect_resonance()
        signature = self.resonance_system._extract_signature(resonance_data)
        
        initial_count = len(self.resonance_system.signature_database)
        self.resonance_system._store_signature(signature)
        
        assert len(self.resonance_system.signature_database) == initial_count + 1
        assert signature['id'] in self.resonance_system.signature_database
    
    def test_recent_signatures_retrieval(self):
        """Test recent signatures retrieval"""
        # Add some test signatures
        for i in range(5):
            resonance_data = self.resonance_system._detect_resonance()
            signature = self.resonance_system._extract_signature(resonance_data)
            self.resonance_system._store_signature(signature)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        recent = self.resonance_system.get_recent_signatures(3)
        assert len(recent) == 3
        
        # Verify they're sorted by timestamp (most recent first)
        timestamps = [sig['timestamp'] for sig in recent]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop functionality"""
        assert not self.resonance_system.monitoring
        
        self.resonance_system.start_monitoring()
        assert self.resonance_system.monitoring
        assert self.resonance_system.capture_thread is not None
        
        time.sleep(0.5)  # Let it run briefly
        
        self.resonance_system.stop_monitoring()
        assert not self.resonance_system.monitoring
    
    def test_visualization_generation(self):
        """Test visualization generation"""
        # Create and store a signature
        resonance_data = self.resonance_system._detect_resonance()
        signature = self.resonance_system._extract_signature(resonance_data)
        self.resonance_system._store_signature(signature)
        
        # Generate visualization
        viz_data = self.resonance_system.generate_visualization(signature['id'])
        assert viz_data is not None
        assert isinstance(viz_data, str)  # JSON string
        
        # Test with non-existent signature
        viz_data_none = self.resonance_system.generate_visualization('nonexistent')
        assert viz_data_none is None

class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.resonance_system = ResonanceCapture()
    
    def test_full_capture_cycle(self):
        """Test complete capture cycle"""
        # Start monitoring
        self.resonance_system.start_monitoring()
        
        # Let it run and capture some signatures
        time.sleep(2.0)
        
        # Stop monitoring
        self.resonance_system.stop_monitoring()
        
        # Verify signatures were captured
        signatures = self.resonance_system.get_recent_signatures()
        assert len(signatures) > 0
        
        # Verify signature quality
        for signature in signatures:
            assert signature['strength'] > 0
            assert len(signature['harmonics']) > 0
            assert 0 <= signature['coherence_factor'] <= 1
    
    def test_concurrent_operations(self):
        """Test concurrent operations"""
        # Start monitoring in background
        self.resonance_system.start_monitoring()
        
        # Perform operations while monitoring
        time.sleep(0.5)
        
        # Get signatures while monitoring is active
        signatures = self.resonance_system.get_recent_signatures()
        
        # Generate visualizations while monitoring
        if signatures:
            viz = self.resonance_system.generate_visualization(signatures[0]['id'])
            assert viz is not None
        
        # Stop monitoring
        self.resonance_system.stop_monitoring()
    
    def test_performance_benchmarks(self):
        """Test system performance benchmarks"""
        # Measure detection time
        start_time = time.time()
        resonance_data = self.resonance_system._detect_resonance()
        detection_time = time.time() - start_time
        
        assert detection_time < 0.1  # Should complete in under 100ms
        
        # Measure signature extraction time
        start_time = time.time()
        signature = self.resonance_system._extract_signature(resonance_data)
        extraction_time = time.time() - start_time
        
        assert extraction_time < 0.5  # Should complete in under 500ms
        
        # Measure visualization generation time
        self.resonance_system._store_signature(signature)
        start_time = time.time()
        viz = self.resonance_system.generate_visualization(signature['id'])
        viz_time = time.time() - start_time
        
        assert viz_time < 1.0  # Should complete in under 1 second

def test_system_constants():
    """Test system constants and configurations"""
    resonance_system = ResonanceCapture()
    
    # Test that detection produces consistent frequency ranges
    frequencies = []
    for _ in range(10):
        data = resonance_system._detect_resonance()
        frequencies.append(data['frequency'])
    
    # Should be around 432 Hz base frequency
    mean_freq = np.mean(frequencies)
    assert 430 < mean_freq < 435
    
    # Should have reasonable variance (quantum fluctuations)
    freq_std = np.std(frequencies)
    assert 0.01 < freq_std < 1.0

if __name__ == "__main__":
    print("Running Resonance Signature Capture System Tests...")
    print("=" * 60)
    
    # Run basic functionality tests
    test_system = TestResonanceCapture()
    test_system.setup_method()
    
    try:
        test_system.test_initialization()
        print("✓ Initialization test passed")
        
        test_system.test_resonance_detection()
        print("✓ Resonance detection test passed")
        
        test_system.test_signature_extraction()
        print("✓ Signature extraction test passed")
        
        test_system.test_harmonic_analysis()
        print("✓ Harmonic analysis test passed")
        
        test_system.test_coherence_calculation()
        print("✓ Coherence calculation test passed")
        
        test_system.test_quantum_signature_generation()
        print("✓ Quantum signature generation test passed")
        
        test_system.test_signature_storage()
        print("✓ Signature storage test passed")
        
        test_system.test_recent_signatures_retrieval()
        print("✓ Recent signatures retrieval test passed")
        
        test_system.test_visualization_generation()
        print("✓ Visualization generation test passed")
        
        print("=" * 60)
        print("🎉 All tests passed! System is operational.")
        print("🌊 Resonance Signature Capture System ready for breakthrough discoveries!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("🔧 Please check system configuration and dependencies")
    
    print("=" * 60)