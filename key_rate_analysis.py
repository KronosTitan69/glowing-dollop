"""
Key Rate Analysis for Quantum Key Distribution

This module implements comprehensive key rate analysis including:
- Key generation rate calculations as functions of channel noise
- Quantum bit error rate (QBER) analysis
- Security parameter calculations
- Throughput vs security trade-offs
- Performance optimization algorithms

Author: Quantum Encryption Verification System
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ChannelParameters:
    """Parameters describing quantum channel characteristics"""
    distance: float  # km
    attenuation: float  # dB/km
    noise_level: float  # probability of error
    detector_efficiency: float  # detector efficiency
    dark_count_rate: float  # dark count rate per second
    dead_time: float  # detector dead time in seconds

@dataclass
class KeyRateResults:
    """Results from key rate analysis"""
    raw_key_rate: float  # bits per second
    sifted_key_rate: float  # bits per second
    final_key_rate: float  # bits per second
    qber: float  # quantum bit error rate
    security_parameter: float  # security parameter
    optimal_parameters: Dict  # optimal system parameters

class KeyRateAnalyzer:
    """
    Comprehensive key rate analyzer for QKD systems.
    
    This class implements various key rate calculation methods including:
    - Asymptotic key rates
    - Finite-key analysis
    - Security parameter optimization
    - Performance trade-off analysis
    """
    
    def __init__(self, channel_params: ChannelParameters):
        self.channel_params = channel_params
        self.results_history = []
        
    def calculate_transmission_probability(self, distance: float) -> float:
        """
        Calculate photon transmission probability through fiber.
        
        Args:
            distance: Transmission distance in km
            
        Returns:
            Transmission probability
        """
        # Exponential attenuation model
        return np.exp(-self.channel_params.attenuation * distance / 10)
    
    def calculate_detection_probability(self, distance: float) -> float:
        """
        Calculate photon detection probability.
        
        Args:
            distance: Transmission distance in km
            
        Returns:
            Detection probability
        """
        transmission_prob = self.calculate_transmission_probability(distance)
        detector_efficiency = self.channel_params.detector_efficiency
        
        return transmission_prob * detector_efficiency
    
    def calculate_dark_count_probability(self, time_window: float) -> float:
        """
        Calculate dark count probability in time window.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Dark count probability
        """
        return self.channel_params.dark_count_rate * time_window
    
    def calculate_asymptotic_key_rate(self, distance: float, 
                                     sifting_efficiency: float = 0.5) -> float:
        """
        Calculate asymptotic key rate for BB84 protocol.
        
        Args:
            distance: Transmission distance in km
            sifting_efficiency: Efficiency of sifting process
            
        Returns:
            Asymptotic key rate in bits per second
        """
        # Detection probability
        p_det = self.calculate_detection_probability(distance)
        
        # Dark count probability (assuming 1 ns time window)
        p_dark = self.calculate_dark_count_probability(1e-9)
        
        # Total detection probability
        p_total = p_det + p_dark - p_det * p_dark
        
        # Sifting efficiency
        p_sift = sifting_efficiency
        
        # Asymptotic key rate (simplified)
        key_rate = p_total * p_sift * 0.5  # 50% for privacy amplification
        
        return key_rate
    
    def calculate_finite_key_rate(self, distance: float, 
                                 num_qubits: int, 
                                 security_parameter: float = 0.1) -> float:
        """
        Calculate finite-key key rate with security analysis.
        
        Args:
            distance: Transmission distance in km
            num_qubits: Number of qubits transmitted
            security_parameter: Security parameter
            
        Returns:
            Finite-key rate in bits per second
        """
        # Asymptotic rate
        asymptotic_rate = self.calculate_asymptotic_key_rate(distance)
        
        # Finite-key correction factor
        # Based on security analysis for finite key lengths
        finite_correction = 1 - 2 * np.sqrt(security_parameter / num_qubits)
        
        return asymptotic_rate * max(0, finite_correction)
    
    def calculate_qber(self, distance: float, 
                      eavesdropping_probability: float = 0.0) -> float:
        """
        Calculate quantum bit error rate.
        
        Args:
            distance: Transmission distance in km
            eavesdropping_probability: Probability of eavesdropping
            
        Returns:
            Quantum bit error rate
        """
        # Base error rate from channel noise
        base_error = self.channel_params.noise_level
        
        # Distance-dependent errors
        distance_error = 1 - self.calculate_transmission_probability(distance)
        
        # Eavesdropping-induced errors
        eavesdropping_error = eavesdropping_probability * 0.25  # 25% error from eavesdropping
        
        # Total QBER
        qber = base_error + distance_error + eavesdropping_error
        
        return min(qber, 0.5)  # Cap at 50%
    
    def calculate_security_parameter(self, qber: float, 
                                   num_qubits: int) -> float:
        """
        Calculate security parameter based on QBER and key length.
        
        Args:
            qber: Quantum bit error rate
            num_qubits: Number of qubits
            
        Returns:
            Security parameter
        """
        # Security threshold (typically 11% for BB84)
        qber_threshold = 0.11
        
        # Security parameter based on QBER
        if qber > qber_threshold:
            return 0.0  # No security
        
        # Finite-key security parameter
        security_param = 1 - qber / qber_threshold
        
        # Additional finite-key correction
        finite_correction = 1 - 2 * np.sqrt(0.1 / num_qubits)
        
        return security_param * max(0, finite_correction)
    
    def optimize_key_rate(self, distance_range: Tuple[float, float], 
                         num_points: int = 50) -> Dict:
        """
        Optimize key rate over distance range.
        
        Args:
            distance_range: (min_distance, max_distance) in km
            num_points: Number of points to evaluate
            
        Returns:
            Dictionary with optimization results
        """
        distances = np.linspace(distance_range[0], distance_range[1], num_points)
        key_rates = []
        qbers = []
        security_params = []
        
        for distance in distances:
            # Calculate key rate
            key_rate = self.calculate_finite_key_rate(distance, 1000)
            key_rates.append(key_rate)
            
            # Calculate QBER
            qber = self.calculate_qber(distance)
            qbers.append(qber)
            
            # Calculate security parameter
            security_param = self.calculate_security_parameter(qber, 1000)
            security_params.append(security_param)
        
        # Find optimal distance
        optimal_idx = np.argmax(key_rates)
        optimal_distance = distances[optimal_idx]
        optimal_key_rate = key_rates[optimal_idx]
        
        return {
            'distances': distances,
            'key_rates': key_rates,
            'qbers': qbers,
            'security_params': security_params,
            'optimal_distance': optimal_distance,
            'optimal_key_rate': optimal_key_rate
        }
    
    def analyze_security_throughput_tradeoff(self, 
                                            distance: float,
                                            eavesdropping_range: Tuple[float, float],
                                            num_points: int = 50) -> Dict:
        """
        Analyze trade-off between security and throughput.
        
        Args:
            distance: Transmission distance in km
            eavesdropping_range: (min_eavesdropping, max_eavesdropping) probability
            num_points: Number of points to evaluate
            
        Returns:
            Dictionary with trade-off analysis
        """
        eavesdropping_probs = np.linspace(eavesdropping_range[0], 
                                        eavesdropping_range[1], 
                                        num_points)
        
        throughputs = []
        security_levels = []
        
        for eaves_prob in eavesdropping_probs:
            # Calculate QBER with eavesdropping
            qber = self.calculate_qber(distance, eaves_prob)
            
            # Calculate key rate
            key_rate = self.calculate_finite_key_rate(distance, 1000)
            
            # Calculate security parameter
            security_param = self.calculate_security_parameter(qber, 1000)
            
            # Effective throughput (security-weighted)
            effective_throughput = key_rate * security_param
            
            throughputs.append(effective_throughput)
            security_levels.append(security_param)
        
        return {
            'eavesdropping_probs': eavesdropping_probs,
            'throughputs': throughputs,
            'security_levels': security_levels,
            'qbers': [self.calculate_qber(distance, ep) for ep in eavesdropping_probs]
        }

def create_key_rate_visualizations(analyzer: KeyRateAnalyzer):
    """
    Create comprehensive visualizations for key rate analysis.
    
    Args:
        analyzer: KeyRateAnalyzer instance
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Key Rate vs Distance
    ax1 = plt.subplot(3, 3, 1)
    distances = np.linspace(0, 100, 100)
    key_rates = [analyzer.calculate_finite_key_rate(d, 1000) for d in distances]
    ax1.plot(distances, key_rates, 'b-', linewidth=2, label='Key Rate')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Key Rate (bits/sec)')
    ax1.set_title('Key Generation Rate vs Distance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. QBER vs Distance
    ax2 = plt.subplot(3, 3, 2)
    qbers = [analyzer.calculate_qber(d) for d in distances]
    ax2.plot(distances, qbers, 'r-', linewidth=2, label='QBER')
    ax2.axhline(y=0.11, color='black', linestyle='--', label='Security Threshold')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('QBER')
    ax2.set_title('Quantum Bit Error Rate vs Distance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Security Parameter vs Distance
    ax3 = plt.subplot(3, 3, 3)
    security_params = [analyzer.calculate_security_parameter(q, 1000) for q in qbers]
    ax3.plot(distances, security_params, 'g-', linewidth=2, label='Security Parameter')
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Security Parameter')
    ax3.set_title('Security Parameter vs Distance')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Key Rate vs Channel Noise
    ax4 = plt.subplot(3, 3, 4)
    noise_levels = np.linspace(0.001, 0.1, 50)
    key_rates_noise = []
    for noise in noise_levels:
        temp_analyzer = KeyRateAnalyzer(ChannelParameters(
            distance=50, attenuation=0.2, noise_level=noise,
            detector_efficiency=0.8, dark_count_rate=1e-6, dead_time=1e-6
        ))
        key_rates_noise.append(temp_analyzer.calculate_finite_key_rate(50, 1000))
    
    ax4.plot(noise_levels, key_rates_noise, 'purple', linewidth=2)
    ax4.set_xlabel('Channel Noise Level')
    ax4.set_ylabel('Key Rate (bits/sec)')
    ax4.set_title('Key Rate vs Channel Noise')
    ax4.grid(True, alpha=0.3)
    
    # 5. Security vs Throughput Trade-off
    ax5 = plt.subplot(3, 3, 5)
    tradeoff = analyzer.analyze_security_throughput_tradeoff(50, (0, 0.5))
    ax5.scatter(tradeoff['security_levels'], tradeoff['throughputs'], 
               c=tradeoff['eavesdropping_probs'], cmap='viridis', s=50)
    ax5.set_xlabel('Security Level')
    ax5.set_ylabel('Effective Throughput')
    ax5.set_title('Security vs Throughput Trade-off')
    ax5.grid(True, alpha=0.3)
    
    # 6. QBER vs Eavesdropping Probability
    ax6 = plt.subplot(3, 3, 6)
    eaves_probs = np.linspace(0, 0.5, 50)
    qbers_eaves = [analyzer.calculate_qber(50, ep) for ep in eaves_probs]
    ax6.plot(eaves_probs, qbers_eaves, 'orange', linewidth=2)
    ax6.axhline(y=0.11, color='black', linestyle='--', label='Security Threshold')
    ax6.set_xlabel('Eavesdropping Probability')
    ax6.set_ylabel('QBER')
    ax6.set_title('QBER vs Eavesdropping Probability')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. Detector Efficiency Impact
    ax7 = plt.subplot(3, 3, 7)
    efficiencies = np.linspace(0.1, 1.0, 50)
    key_rates_eff = []
    for eff in efficiencies:
        temp_analyzer = KeyRateAnalyzer(ChannelParameters(
            distance=50, attenuation=0.2, noise_level=0.01,
            detector_efficiency=eff, dark_count_rate=1e-6, dead_time=1e-6
        ))
        key_rates_eff.append(temp_analyzer.calculate_finite_key_rate(50, 1000))
    
    ax7.plot(efficiencies, key_rates_eff, 'brown', linewidth=2)
    ax7.set_xlabel('Detector Efficiency')
    ax7.set_ylabel('Key Rate (bits/sec)')
    ax7.set_title('Key Rate vs Detector Efficiency')
    ax7.grid(True, alpha=0.3)
    
    # 8. Distance vs Security Heatmap
    ax8 = plt.subplot(3, 3, 8)
    distances_2d = np.linspace(0, 100, 20)
    eaves_2d = np.linspace(0, 0.5, 20)
    security_matrix = np.zeros((20, 20))
    
    for i, d in enumerate(distances_2d):
        for j, e in enumerate(eaves_2d):
            qber = analyzer.calculate_qber(d, e)
            security_matrix[i, j] = analyzer.calculate_security_parameter(qber, 1000)
    
    im = ax8.imshow(security_matrix, extent=[0, 0.5, 0, 100], aspect='auto', cmap='RdYlGn')
    ax8.set_xlabel('Eavesdropping Probability')
    ax8.set_ylabel('Distance (km)')
    ax8.set_title('Security Parameter Heatmap')
    plt.colorbar(im, ax=ax8, label='Security Parameter')
    
    # 9. Performance Summary
    ax9 = plt.subplot(3, 3, 9)
    # Create a summary table
    summary_data = {
        'Metric': ['Max Key Rate', 'Min QBER', 'Max Security', 'Optimal Distance'],
        'Value': [
            f"{max(key_rates):.2e}",
            f"{min(qbers):.4f}",
            f"{max(security_params):.4f}",
            f"{distances[np.argmax(key_rates)]:.1f} km"
        ]
    }
    
    ax9.axis('tight')
    ax9.axis('off')
    table = ax9.table(cellText=list(zip(summary_data['Metric'], summary_data['Value'])),
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax9.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('quantum_encryption_verification/key_rate_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def benchmark_against_literature():
    """
    Benchmark key rate calculations against literature values.
    
    Returns:
        Dictionary with benchmark results
    """
    # Literature benchmarks (approximate values)
    literature_benchmarks = {
        'BB84_10km': {'distance': 10, 'key_rate': 1e6, 'qber': 0.02},
        'BB84_50km': {'distance': 50, 'key_rate': 1e4, 'qber': 0.05},
        'BB84_100km': {'distance': 100, 'key_rate': 1e2, 'qber': 0.08}
    }
    
    # Create analyzer with realistic parameters
    channel_params = ChannelParameters(
        distance=50, attenuation=0.2, noise_level=0.01,
        detector_efficiency=0.8, dark_count_rate=1e-6, dead_time=1e-6
    )
    
    analyzer = KeyRateAnalyzer(channel_params)
    
    benchmark_results = {}
    
    for benchmark_name, benchmark_data in literature_benchmarks.items():
        distance = benchmark_data['distance']
        literature_key_rate = benchmark_data['key_rate']
        literature_qber = benchmark_data['qber']
        
        # Calculate our results
        our_key_rate = analyzer.calculate_finite_key_rate(distance, 1000)
        our_qber = analyzer.calculate_qber(distance)
        
        # Calculate relative errors
        key_rate_error = abs(our_key_rate - literature_key_rate) / literature_key_rate
        qber_error = abs(our_qber - literature_qber) / literature_qber
        
        benchmark_results[benchmark_name] = {
            'literature_key_rate': literature_key_rate,
            'our_key_rate': our_key_rate,
            'key_rate_error': key_rate_error,
            'literature_qber': literature_qber,
            'our_qber': our_qber,
            'qber_error': qber_error
        }
    
    return benchmark_results

if __name__ == "__main__":
    print("Key Rate Analysis for Quantum Key Distribution")
    print("=" * 50)
    
    # Create analyzer with realistic parameters
    channel_params = ChannelParameters(
        distance=50, attenuation=0.2, noise_level=0.01,
        detector_efficiency=0.8, dark_count_rate=1e-6, dead_time=1e-6
    )
    
    analyzer = KeyRateAnalyzer(channel_params)
    
    # Run optimization
    print("Running key rate optimization...")
    optimization_results = analyzer.optimize_key_rate((0, 100), 100)
    
    print(f"Optimal distance: {optimization_results['optimal_distance']:.2f} km")
    print(f"Optimal key rate: {optimization_results['optimal_key_rate']:.2e} bits/sec")
    
    # Create visualizations
    print("Creating visualizations...")
    create_key_rate_visualizations(analyzer)
    
    # Run benchmarks
    print("Running literature benchmarks...")
    benchmark_results = benchmark_against_literature()
    
    print("\nBenchmark Results:")
    print("-" * 30)
    for benchmark_name, results in benchmark_results.items():
        print(f"{benchmark_name}:")
        print(f"  Key Rate Error: {results['key_rate_error']:.2%}")
        print(f"  QBER Error: {results['qber_error']:.2%}")
        print()


