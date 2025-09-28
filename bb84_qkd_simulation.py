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
    from qiskit.visualization import plot_bloch_multivector, circuit_drawer
    from qiskit.providers.basic_provider import BasicProvider
    from qiskit import execute, transpile
    from qiskit.quantum_info import state_fidelity
    QISKIT_AVAILABLE = True
    print("✓ Qiskit successfully imported - quantum circuit simulation enabled")
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Using classical probability simulation.")

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
        
        # Qiskit integration
        self.quantum_backend = None
        self.quantum_circuits = []
        self.use_quantum_circuits = QISKIT_AVAILABLE
        
        if QISKIT_AVAILABLE:
            try:
                provider = BasicProvider()
                self.quantum_backend = provider.get_backend('basic_simulator')
                print(f"✓ Quantum backend initialized: {self.quantum_backend.name()}")
            except Exception as e:
                print(f"Warning: Could not initialize quantum backend: {e}")
                self.use_quantum_circuits = False
        
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
    
    def create_alice_quantum_circuit(self, bit: int, basis: Basis) -> 'QuantumCircuit':
        """
        Create quantum circuit for Alice's qubit preparation using Qiskit.
        
        Args:
            bit: 0 or 1 - the bit Alice wants to encode
            basis: Z or X - the basis Alice chooses
            
        Returns:
            QuantumCircuit for Alice's preparation (or None if Qiskit unavailable)
        """
        if not self.use_quantum_circuits:
            return None
            
        # Create quantum circuit with 1 qubit
        qc = QuantumCircuit(1, 1)
        
        # Step 1: Initialize qubit based on bit value
        if bit == 1:
            qc.x(0)  # Apply X gate for |1⟩ state
        
        # Step 2: Apply basis transformation
        if basis == Basis.X:
            qc.h(0)  # Apply Hadamard for X basis (|+⟩ or |-⟩)
        
        qc.barrier()  # Visual separation
        
        return qc
    
    def create_bob_measurement_circuit(self, alice_circuit: 'QuantumCircuit', 
                                     measurement_basis: Basis) -> 'QuantumCircuit':
        """
        Create complete quantum circuit including Bob's measurement.
        
        Args:
            alice_circuit: Alice's preparation circuit
            measurement_basis: Bob's chosen measurement basis
            
        Returns:
            Complete circuit with measurement (or None if Qiskit unavailable)
        """
        if not self.use_quantum_circuits or alice_circuit is None:
            return None
            
        # Copy Alice's circuit
        qc = alice_circuit.copy()
        
        # Apply basis transformation for measurement
        if measurement_basis == Basis.X:
            qc.h(0)  # Hadamard before measurement for X basis
        
        qc.barrier()
        qc.measure(0, 0)  # Measure qubit 0 into classical bit 0
        
        return qc
    
    def simulate_quantum_channel_noise(self, circuit: 'QuantumCircuit') -> 'QuantumCircuit':
        """
        Add quantum channel noise to the circuit.
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            Circuit with noise applied (or original if Qiskit unavailable)
        """
        if not self.use_quantum_circuits or circuit is None:
            return circuit
            
        noisy_circuit = circuit.copy()
        
        # Add noise based on channel_noise parameter
        if self.params.channel_noise > 0:
            # Add depolarizing noise (simplified)
            if random.random() < self.params.channel_noise:
                # Apply random Pauli gate for noise
                noise_gate = random.choice(['x', 'z', 'y'])
                if noise_gate == 'x':
                    noisy_circuit.x(0)
                elif noise_gate == 'z':
                    noisy_circuit.z(0)
                else:  # y gate
                    noisy_circuit.y(0)
        
        return noisy_circuit
    
    def execute_quantum_circuit(self, circuit: 'QuantumCircuit', shots: int = 1) -> Tuple[List[int], Dict]:
        """
        Execute quantum circuit and return measurement results.
        
        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots
            
        Returns:
            Tuple of (measurements, circuit_info)
        """
        if not self.use_quantum_circuits or circuit is None:
            # Fallback to classical simulation
            return [random.randint(0, 1)], {'method': 'classical_fallback'}
            
        try:
            # Execute circuit
            job = execute(circuit, self.quantum_backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            # Convert counts to measurement list
            measurements = []
            for outcome, count in counts.items():
                measurements.extend([int(outcome)] * count)
            
            circuit_info = {
                'method': 'quantum_circuit',
                'shots': shots,
                'counts': counts,
                'circuit_depth': circuit.depth(),
                'num_ops': len(circuit.data)
            }
            
            return measurements, circuit_info
            
        except Exception as e:
            print(f"Warning: Quantum circuit execution failed: {e}")
            # Fallback to classical simulation
            return [random.randint(0, 1)], {'method': 'quantum_fallback', 'error': str(e)}
    
    def quantum_simulate_bb84_exchange(self, alice_bit: int, alice_basis: Basis, 
                                     bob_basis: Basis) -> Tuple[int, Dict]:
        """
        Simulate complete BB84 qubit exchange using quantum circuits.
        
        Args:
            alice_bit: Alice's bit (0 or 1)
            alice_basis: Alice's preparation basis
            bob_basis: Bob's measurement basis
            
        Returns:
            Tuple of (bob_measurement, execution_info)
        """
        if not self.use_quantum_circuits:
            # Fallback to classical simulation with same logic as original methods
            if alice_basis == bob_basis:
                # Same basis - should get same result (with noise)
                if random.random() < self.params.channel_noise:
                    measurement = 1 - alice_bit  # Bit flip due to noise
                else:
                    measurement = alice_bit
            else:
                # Different basis - random result
                measurement = random.randint(0, 1)
            
            return measurement, {'method': 'classical_simulation'}
        
        # Step 1: Alice prepares qubit
        alice_circuit = self.create_alice_quantum_circuit(alice_bit, alice_basis)
        
        # Step 2: Apply channel noise
        noisy_circuit = self.simulate_quantum_channel_noise(alice_circuit)
        
        # Step 3: Bob measures
        complete_circuit = self.create_bob_measurement_circuit(noisy_circuit, bob_basis)
        
        # Step 4: Execute circuit
        measurements, circuit_info = self.execute_quantum_circuit(complete_circuit, shots=1)
        
        bob_measurement = measurements[0] if measurements else 0
        
        # Store circuit for potential visualization
        self.quantum_circuits.append({
            'alice_bit': alice_bit,
            'alice_basis': alice_basis.name,
            'bob_basis': bob_basis.name,
            'bob_measurement': bob_measurement,
            'circuit': complete_circuit,
            'execution_info': circuit_info
        })
        
        return bob_measurement, circuit_info
    
    def run_quantum_enhanced_protocol(self) -> QKDResults:
        """
        Run BB84 protocol with quantum circuit simulation when available.
        
        Returns:
            QKDResults object with quantum-enhanced simulation results
        """
        print(f"Running {'quantum-enhanced' if self.use_quantum_circuits else 'classical'} BB84 protocol...")
        
        # Clear previous results
        self.alice_bits = []
        self.alice_bases = []
        self.bob_bases = []
        self.bob_measurements = []
        self.quantum_circuits = []
        
        # Generate and process qubits
        for i in range(self.params.num_qubits):
            # Alice's random choices
            alice_bit = random.randint(0, 1)
            alice_basis = random.choice([Basis.Z, Basis.X])
            
            # Bob's random basis choice
            bob_basis = random.choice([Basis.Z, Basis.X])
            
            # Simulate quantum exchange (uses quantum circuits if available)
            bob_measurement, exec_info = self.quantum_simulate_bb84_exchange(
                alice_bit, alice_basis, bob_basis
            )
            
            # Apply detector efficiency
            if random.random() > self.params.detector_efficiency:
                bob_measurement = None  # No detection
            
            # Store results
            self.alice_bits.append(alice_bit)
            self.alice_bases.append(alice_basis)
            self.bob_bases.append(bob_basis)
            self.bob_measurements.append(bob_measurement)
            
            # Progress indicator for large simulations
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{self.params.num_qubits} qubits")
        
        # Classical sifting
        alice_sifted, bob_sifted, sifting_efficiency = self.classical_sifting()
        
        # Calculate error rate
        qber = self.calculate_error_rate(alice_sifted, bob_sifted)
        
        # Security analysis
        eavesdropping_detected, security_parameter = self.security_analysis(qber)
        
        # Privacy amplification (simplified)
        final_key_length = int(len(alice_sifted) * security_parameter * 0.5)
        
        # Calculate key generation rate
        key_generation_rate = final_key_length / self.params.num_qubits
        
        print(f"✓ {'Quantum-enhanced' if self.use_quantum_circuits else 'Classical'} BB84 protocol completed:")
        print(f"  Raw key length: {len(alice_sifted)}")
        print(f"  QBER: {qber:.4f}")
        print(f"  Security parameter: {security_parameter:.4f}")
        print(f"  Final key length: {final_key_length}")
        print(f"  Key generation rate: {key_generation_rate:.4f}")
        
        return QKDResults(
            raw_key_length=len(alice_sifted),
            sifted_key_length=len(alice_sifted),
            final_key_length=final_key_length,
            error_rate=qber,
            key_generation_rate=key_generation_rate,
            security_parameter=security_parameter,
            eavesdropping_detected=eavesdropping_detected
        )
    
    def get_quantum_circuit_statistics(self) -> Dict:
        """
        Get statistics about quantum circuit executions.
        
        Returns:
            Dictionary with quantum circuit statistics
        """
        if not self.quantum_circuits:
            return {'message': 'No quantum circuits executed'}
        
        stats = {
            'total_circuits': len(self.quantum_circuits),
            'quantum_executions': sum(1 for c in self.quantum_circuits 
                                    if c['execution_info']['method'] == 'quantum_circuit'),
            'classical_fallbacks': sum(1 for c in self.quantum_circuits 
                                     if 'fallback' in c['execution_info']['method']),
            'average_circuit_depth': 0,
            'total_quantum_operations': 0
        }
        
        quantum_circuits = [c for c in self.quantum_circuits 
                          if c['execution_info']['method'] == 'quantum_circuit']
        
        if quantum_circuits:
            stats['average_circuit_depth'] = sum(c['execution_info']['circuit_depth'] 
                                               for c in quantum_circuits) / len(quantum_circuits)
            stats['total_quantum_operations'] = sum(c['execution_info']['num_ops'] 
                                                  for c in quantum_circuits)
        
        return stats
    
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

def simulate_quantum_enhanced_scenarios():
    """
    Run QKD simulations with quantum circuit enhancement.
    
    Returns:
        Dictionary of results comparing classical and quantum simulations
    """
    print("Quantum-Enhanced BB84 QKD Protocol Simulation")
    print("=" * 50)
    
    scenarios = {
        'quantum_ideal': QKDParameters(
            num_qubits=500,  # Smaller for quantum simulation
            channel_noise=0.001,
            detector_efficiency=0.95,
            eavesdropping_probability=0.0
        ),
        'quantum_realistic': QKDParameters(
            num_qubits=500,
            channel_noise=0.02,
            detector_efficiency=0.8,
            eavesdropping_probability=0.0
        ),
        'quantum_noisy': QKDParameters(
            num_qubits=300,
            channel_noise=0.05,
            detector_efficiency=0.7,
            eavesdropping_probability=0.0
        )
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\nRunning {scenario_name} scenario...")
        qkd = BB84QKD(params)
        
        # Run quantum-enhanced protocol
        result = qkd.run_quantum_enhanced_protocol()
        results[scenario_name] = result
        
        # Get quantum circuit statistics
        quantum_stats = qkd.get_quantum_circuit_statistics()
        
        print(f"  QBER: {result.error_rate:.4f}")
        print(f"  Key Generation Rate: {result.key_generation_rate:.4f}")
        print(f"  Eavesdropping Detected: {result.eavesdropping_detected}")
        
        if 'total_circuits' in quantum_stats:
            print(f"  Quantum Circuits: {quantum_stats['total_circuits']}")
            if quantum_stats['quantum_executions'] > 0:
                print(f"  Quantum Executions: {quantum_stats['quantum_executions']}")
                print(f"  Avg Circuit Depth: {quantum_stats['average_circuit_depth']:.1f}")
        
    return results

def compare_classical_vs_quantum_simulation():
    """
    Compare classical probability simulation vs quantum circuit simulation.
    
    Returns:
        Comparison results
    """
    print("\nComparison: Classical vs Quantum Simulation")
    print("=" * 45)
    
    # Test parameters
    params = QKDParameters(
        num_qubits=200,
        channel_noise=0.02,
        detector_efficiency=0.9,
        eavesdropping_probability=0.0
    )
    
    # Classical simulation
    print("Running classical simulation...")
    qkd_classical = BB84QKD(params)
    qkd_classical.use_quantum_circuits = False  # Force classical
    classical_result = qkd_classical.run_protocol()
    
    # Quantum simulation
    print("Running quantum-enhanced simulation...")
    qkd_quantum = BB84QKD(params)
    quantum_result = qkd_quantum.run_quantum_enhanced_protocol()
    quantum_stats = qkd_quantum.get_quantum_circuit_statistics()
    
    # Compare results
    print("\nComparison Results:")
    print("-" * 30)
    print(f"{'Metric':<25} {'Classical':<12} {'Quantum':<12}")
    print("-" * 30)
    print(f"{'QBER':<25} {classical_result.error_rate:<12.4f} {quantum_result.error_rate:<12.4f}")
    print(f"{'Key Rate':<25} {classical_result.key_generation_rate:<12.4f} {quantum_result.key_generation_rate:<12.4f}")
    print(f"{'Security Param':<25} {classical_result.security_parameter:<12.4f} {quantum_result.security_parameter:<12.4f}")
    print(f"{'Final Key Length':<25} {classical_result.final_key_length:<12} {quantum_result.final_key_length:<12}")
    
    if 'total_circuits' in quantum_stats:
        print(f"\nQuantum Circuit Statistics:")
        print(f"  Total circuits executed: {quantum_stats['total_circuits']}")
        print(f"  Quantum executions: {quantum_stats.get('quantum_executions', 0)}")
        print(f"  Classical fallbacks: {quantum_stats.get('classical_fallbacks', 0)}")
    
    return {
        'classical': classical_result,
        'quantum': quantum_result,
        'quantum_stats': quantum_stats
    }

if __name__ == "__main__":
    print("BB84 Quantum Key Distribution Protocol Simulation")
    print("=" * 50)
    
    # Check Qiskit availability
    if QISKIT_AVAILABLE:
        print("🚀 Qiskit detected - running enhanced quantum simulation")
        
        # Run quantum-enhanced scenarios
        quantum_results = simulate_quantum_enhanced_scenarios()
        
        # Run comparison
        comparison_results = compare_classical_vs_quantum_simulation()
        
        print("\n" + "=" * 50)
        print("QUANTUM-ENHANCED SIMULATION COMPLETED")
        print("=" * 50)
        
    else:
        print("📊 Qiskit not available - running classical simulation")
        
        # Run classical simulation scenarios
        results = simulate_qkd_scenarios()
        
        print("\n" + "=" * 50)
        print("CLASSICAL SIMULATION COMPLETED")
        print("=" * 50)
    
    # Run original simulation scenarios for comparison
    print("\nRunning original scenarios for reference...")
    original_results = simulate_qkd_scenarios()
    
    # Print final summary
    print("\nFinal Simulation Summary:")
    print("-" * 30)
    for scenario, result in original_results.items():
        print(f"{scenario:20} | QBER: {result.error_rate:.4f} | "
              f"Key Rate: {result.key_generation_rate:.4f} | "
              f"Secure: {not result.eavesdropping_detected}")
    
    print(f"\n✓ BB84 simulation completed with {'quantum circuits' if QISKIT_AVAILABLE else 'classical methods'}")


