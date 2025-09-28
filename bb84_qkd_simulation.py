"""
BB84 Quantum Key Distribution Protocol Simulation

This module implements a comprehensive simulation of the BB84 QKD protocol,
including Alice's qubit preparation, Bob's measurement, and Eve's eavesdropping.
The simulation includes error rate analysis, key generation rate calculations,
and security analysis based on quantum mechanics principles.

Author: Quantum Encryption Verification System
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import random
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress Qiskit warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector, random_statevector
    from qiskit.visualization import plot_bloch_multivector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Some quantum visualizations will be disabled.")

class Basis(Enum):
    """Quantum measurement bases"""
    Z = 0  # Computational basis (|0⟩, |1⟩)
    X = 1  # Hadamard basis (|+⟩, |-⟩)

class QubitState(Enum):
    """Qubit states in different bases"""
    ZERO = 0      # |0⟩ in Z basis
    ONE = 1       # |1⟩ in Z basis
    PLUS = 2       # |+⟩ in X basis
    MINUS = 3      # |-⟩ in X basis

@dataclass
class QKDParameters:
    """Parameters for QKD simulation"""
    num_qubits: int = 1000
    channel_noise: float = 0.01
    detector_efficiency: float = 0.8
    dark_count_rate: float = 1e-6
    eavesdropping_probability: float = 0.0
    sifting_efficiency: float = 0.5

@dataclass
class QKDResults:
    """Results from QKD simulation"""
    raw_key_length: int
    sifted_key_length: int
    final_key_length: int
    error_rate: float
    key_generation_rate: float
    security_parameter: float
    eavesdropping_detected: bool

class BB84QKD:
    """
    BB84 Quantum Key Distribution Protocol Implementation
    
    This class implements the complete BB84 protocol including:
    - Alice's random bit and basis selection
    - Qubit preparation and transmission
    - Bob's random basis selection and measurement
    - Classical sifting and error correction
    - Security analysis and eavesdropping detection
    """
    
    def __init__(self, parameters: QKDParameters):
        self.params = parameters
        self.alice_bits = []
        self.alice_bases = []
        self.bob_bases = []
        self.bob_measurements = []
        self.sifted_key = []
        self.error_positions = []
        
    def alice_prepare_qubits(self) -> List[Tuple[int, Basis]]:
        """
        Alice prepares qubits by randomly selecting bits and bases.
        
        Returns:
            List of (bit, basis) tuples representing Alice's preparation
        """
        self.alice_bits = []
        self.alice_bases = []
        
        for _ in range(self.params.num_qubits):
            # Random bit selection
            bit = random.randint(0, 1)
            # Random basis selection
            basis = random.choice([Basis.Z, Basis.X])
            
            self.alice_bits.append(bit)
            self.alice_bases.append(basis)
        
        return list(zip(self.alice_bits, self.alice_bases))
    
    def apply_channel_noise(self, qubit_state: Tuple[int, Basis]) -> Tuple[int, Basis]:
        """
        Simulate channel noise effects on qubit transmission.
        
        Args:
            qubit_state: (bit, basis) tuple
            
        Returns:
            Modified qubit state after noise
        """
        bit, basis = qubit_state
        
        # Simulate bit flip with probability channel_noise
        if random.random() < self.params.channel_noise:
            bit = 1 - bit
            
        # Simulate basis flip with probability channel_noise/2
        if random.random() < self.params.channel_noise / 2:
            basis = Basis.X if basis == Basis.Z else Basis.Z
            
        return (bit, basis)
    
    def eve_intercept_resend(self, qubit_state: Tuple[int, Basis]) -> Tuple[int, Basis]:
        """
        Eve performs intercept-resend attack.
        
        Args:
            qubit_state: (bit, basis) tuple from Alice
            
        Returns:
            Modified qubit state after Eve's attack
        """
        if random.random() > self.params.eavesdropping_probability:
            return qubit_state
            
        bit, basis = qubit_state
        
        # Eve randomly chooses a basis for measurement
        eve_basis = random.choice([Basis.Z, Basis.X])
        
        # Eve measures in her chosen basis
        if eve_basis == basis:
            # Same basis - Eve gets correct result
            eve_result = bit
        else:
            # Different basis - Eve gets random result
            eve_result = random.randint(0, 1)
        
        # Eve resends in her chosen basis
        return (eve_result, eve_basis)
    
    def bob_measure_qubits(self, alice_preparations: List[Tuple[int, Basis]]) -> List[int]:
        """
        Bob randomly selects bases and measures qubits.
        
        Args:
            alice_preparations: List of (bit, basis) from Alice
            
        Returns:
            List of Bob's measurement results
        """
        self.bob_bases = []
        self.bob_measurements = []
        
        for i, (alice_bit, alice_basis) in enumerate(alice_preparations):
            # Apply channel noise
            noisy_qubit = self.apply_channel_noise((alice_bit, alice_basis))
            
            # Apply Eve's attack
            attacked_qubit = self.eve_intercept_resend(noisy_qubit)
            
            # Bob randomly chooses measurement basis
            bob_basis = random.choice([Basis.Z, Basis.X])
            self.bob_bases.append(bob_basis)
            
            # Bob measures
            if bob_basis == attacked_qubit[1]:
                # Same basis - Bob gets correct result
                measurement = attacked_qubit[0]
            else:
                # Different basis - Bob gets random result
                measurement = random.randint(0, 1)
            
            # Apply detector efficiency
            if random.random() > self.params.detector_efficiency:
                measurement = None  # No detection
                
            self.bob_measurements.append(measurement)
        
        return self.bob_measurements
    
    def classical_sifting(self) -> Tuple[List[int], List[int], float]:
        """
        Perform classical sifting to discard mismatched basis measurements.
        
        Returns:
            Tuple of (Alice's sifted bits, Bob's sifted bits, sifting efficiency)
        """
        alice_sifted = []
        bob_sifted = []
        
        for i in range(len(self.alice_bits)):
            if (self.bob_measurements[i] is not None and 
                self.alice_bases[i] == self.bob_bases[i]):
                alice_sifted.append(self.alice_bits[i])
                bob_sifted.append(self.bob_measurements[i])
        
        sifting_efficiency = len(alice_sifted) / len(self.alice_bits)
        return alice_sifted, bob_sifted, sifting_efficiency
    
    def calculate_error_rate(self, alice_sifted: List[int], bob_sifted: List[int]) -> float:
        """
        Calculate quantum bit error rate (QBER).
        
        Args:
            alice_sifted: Alice's sifted bits
            bob_sifted: Bob's sifted bits
            
        Returns:
            Quantum bit error rate
        """
        if len(alice_sifted) == 0:
            return 0.0
            
        errors = sum(1 for a, b in zip(alice_sifted, bob_sifted) if a != b)
        return errors / len(alice_sifted)
    
    def security_analysis(self, qber: float) -> Tuple[bool, float]:
        """
        Perform security analysis based on QBER.
        
        Args:
            qber: Quantum bit error rate
            
        Returns:
            Tuple of (eavesdropping_detected, security_parameter)
        """
        # Theoretical QBER threshold for security (typically 11%)
        qber_threshold = 0.11
        
        # Security parameter based on QBER
        security_parameter = max(0, 1 - qber / qber_threshold)
        
        # Detect eavesdropping if QBER exceeds threshold
        eavesdropping_detected = qber > qber_threshold
        
        return eavesdropping_detected, security_parameter
    
    def run_protocol(self) -> QKDResults:
        """
        Run the complete BB84 protocol.
        
        Returns:
            QKDResults object with all simulation results
        """
        # Step 1: Alice prepares qubits
        alice_preparations = self.alice_prepare_qubits()
        
        # Step 2: Bob measures qubits
        bob_measurements = self.bob_measure_qubits(alice_preparations)
        
        # Step 3: Classical sifting
        alice_sifted, bob_sifted, sifting_efficiency = self.classical_sifting()
        
        # Step 4: Calculate error rate
        qber = self.calculate_error_rate(alice_sifted, bob_sifted)
        
        # Step 5: Security analysis
        eavesdropping_detected, security_parameter = self.security_analysis(qber)
        
        # Step 6: Privacy amplification (simplified)
        # In practice, this would involve more complex procedures
        final_key_length = int(len(alice_sifted) * security_parameter * 0.5)
        
        # Calculate key generation rate
        key_generation_rate = final_key_length / self.params.num_qubits
        
        return QKDResults(
            raw_key_length=len(alice_sifted),
            sifted_key_length=len(alice_sifted),
            final_key_length=final_key_length,
            error_rate=qber,
            key_generation_rate=key_generation_rate,
            security_parameter=security_parameter,
            eavesdropping_detected=eavesdropping_detected
        )

def simulate_qkd_scenarios():
    """
    Run QKD simulations across different scenarios.
    
    Returns:
        Dictionary of results for different parameter sets
    """
    scenarios = {
        'ideal': QKDParameters(
            num_qubits=1000,
            channel_noise=0.001,
            detector_efficiency=0.95,
            eavesdropping_probability=0.0
        ),
        'realistic': QKDParameters(
            num_qubits=1000,
            channel_noise=0.01,
            detector_efficiency=0.8,
            eavesdropping_probability=0.0
        ),
        'eavesdropping_light': QKDParameters(
            num_qubits=1000,
            channel_noise=0.01,
            detector_efficiency=0.8,
            eavesdropping_probability=0.1
        ),
        'eavesdropping_heavy': QKDParameters(
            num_qubits=1000,
            channel_noise=0.01,
            detector_efficiency=0.8,
            eavesdropping_probability=0.3
        )
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"Running {scenario_name} scenario...")
        qkd = BB84QKD(params)
        result = qkd.run_protocol()
        results[scenario_name] = result
        
        print(f"  QBER: {result.error_rate:.4f}")
        print(f"  Key Generation Rate: {result.key_generation_rate:.4f}")
        print(f"  Eavesdropping Detected: {result.eavesdropping_detected}")
        print()
    
    return results

if __name__ == "__main__":
    print("BB84 Quantum Key Distribution Protocol Simulation")
    print("=" * 50)
    
    # Run simulation scenarios
    results = simulate_qkd_scenarios()
    
    # Print summary
    print("\nSimulation Summary:")
    print("-" * 30)
    for scenario, result in results.items():
        print(f"{scenario:20} | QBER: {result.error_rate:.4f} | "
              f"Key Rate: {result.key_generation_rate:.4f} | "
              f"Secure: {not result.eavesdropping_detected}")


