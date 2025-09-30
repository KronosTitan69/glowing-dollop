"""
Qiskit Integration Module for Quantum Encryption Verification System

This module provides enhanced quantum operations using Qiskit, including:
- Real quantum circuit implementations of BB84 protocol
- Quantum state preparations and measurements
- Noise model simulations
- Circuit visualizations
- Educational quantum demonstrations

Author: Quantum Encryption Verification System
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import random
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Qiskit integration with fallback
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector, random_statevector
    from qiskit.visualization import plot_histogram, circuit_drawer
    from qiskit.providers.basic_provider import BasicProvider
    from qiskit import transpile, execute
    from qiskit.quantum_info import state_fidelity
    QISKIT_AVAILABLE = True
    print("✓ Qiskit successfully imported and available")
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Using fallback quantum simulation.")

# Import from existing BB84 implementation
try:
    from bb84_qkd_simulation import Basis, QKDParameters, QKDResults
except ImportError:
    # Define fallback classes if import fails
    class Basis(Enum):
        Z = 0  # Computational basis
        X = 1  # Hadamard basis
    
    @dataclass
    class QKDParameters:
        num_qubits: int = 1000
        channel_noise: float = 0.01
        detector_efficiency: float = 0.8
        dark_count_rate: float = 1e-6
        eavesdropping_probability: float = 0.0
        sifting_efficiency: float = 0.5

@dataclass
class QuantumCircuitResult:
    """Results from quantum circuit execution"""
    measurements: List[int]
    circuit: Any  # QuantumCircuit or fallback
    probabilities: Dict[str, float]
    fidelity: float

class QiskitBB84Simulator:
    """
    Qiskit-enhanced BB84 QKD Protocol Implementation
    
    This class provides real quantum circuit implementations of the BB84 protocol,
    replacing classical probability calculations with actual quantum circuits.
    """
    
    def __init__(self, parameters: QKDParameters):
        self.params = parameters
        self.backend = None
        self.circuits = []
        self.results = []
        
        if QISKIT_AVAILABLE:
            # Use BasicProvider for local simulation
            provider = BasicProvider()
            self.backend = provider.get_backend('basic_simulator')
            print(f"✓ Using Qiskit backend: {self.backend.name()}")
        else:
            print("⚠ Using fallback classical simulation")
    
    def create_alice_preparation_circuit(self, bit: int, basis: Basis) -> QuantumCircuit:
        """
        Create quantum circuit for Alice's qubit preparation.
        
        Args:
            bit: 0 or 1 - the bit Alice wants to encode
            basis: Z or X - the basis Alice chooses
            
        Returns:
            QuantumCircuit for Alice's preparation
        """
        if not QISKIT_AVAILABLE:
            return self._fallback_circuit(bit, basis)
        
        # Create quantum circuit with 1 qubit and 1 classical bit
        qc = QuantumCircuit(1, 1)
        
        # Step 1: Initialize qubit based on bit value
        if bit == 1:
            qc.x(0)  # Apply X gate for |1⟩ state
        
        # Step 2: Apply basis transformation
        if basis == Basis.X:
            qc.h(0)  # Apply Hadamard for X basis (|+⟩ or |-⟩)
        
        qc.barrier()  # Visual separation
        
        return qc
    
    def create_bob_measurement_circuit(self, preparation_circuit: QuantumCircuit, 
                                     measurement_basis: Basis) -> QuantumCircuit:
        """
        Create quantum circuit for Bob's measurement.
        
        Args:
            preparation_circuit: Alice's preparation circuit
            measurement_basis: Bob's chosen measurement basis
            
        Returns:
            Complete circuit including measurement
        """
        if not QISKIT_AVAILABLE:
            return self._fallback_circuit(0, measurement_basis)
        
        # Copy Alice's circuit
        qc = preparation_circuit.copy()
        
        # Apply basis transformation for measurement
        if measurement_basis == Basis.X:
            qc.h(0)  # Hadamard before measurement for X basis
        
        qc.barrier()
        qc.measure(0, 0)  # Measure qubit 0 into classical bit 0
        
        return qc
    
    def simulate_quantum_channel_noise(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Add quantum channel noise to the circuit.
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            Circuit with noise applied
        """
        if not QISKIT_AVAILABLE:
            return circuit
        
        noisy_circuit = circuit.copy()
        
        # Add noise based on channel_noise parameter
        if self.params.channel_noise > 0:
            # Add depolarizing noise (simplified)
            if random.random() < self.params.channel_noise:
                # Apply random Pauli gate
                noise_gate = random.choice(['x', 'z'])
                if noise_gate == 'x':
                    noisy_circuit.x(0)
                else:
                    noisy_circuit.z(0)
        
        return noisy_circuit
    
    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> QuantumCircuitResult:
        """
        Execute quantum circuit and return results.
        
        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots
            
        Returns:
            QuantumCircuitResult with execution data
        """
        if not QISKIT_AVAILABLE:
            return self._fallback_execution(circuit, shots)
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert counts to probabilities
        probabilities = {state: count/shots for state, count in counts.items()}
        
        # Get most likely measurement (for single shot simulation)
        most_likely_state = max(counts.keys(), key=lambda x: counts[x])
        measurement = int(most_likely_state)
        
        # Calculate fidelity (simplified)
        fidelity = max(probabilities.values()) if probabilities else 0.0
        
        return QuantumCircuitResult(
            measurements=[measurement],
            circuit=circuit,
            probabilities=probabilities,
            fidelity=fidelity
        )
    
    def simulate_bb84_qubit_exchange(self, alice_bit: int, alice_basis: Basis,
                                   bob_basis: Basis) -> Tuple[int, QuantumCircuitResult]:
        """
        Simulate complete BB84 qubit exchange using quantum circuits.
        
        Args:
            alice_bit: Alice's bit (0 or 1)
            alice_basis: Alice's preparation basis
            bob_basis: Bob's measurement basis
            
        Returns:
            Tuple of (bob_measurement, circuit_result)
        """
        # Step 1: Alice prepares qubit
        alice_circuit = self.create_alice_preparation_circuit(alice_bit, alice_basis)
        
        # Step 2: Apply channel noise
        noisy_circuit = self.simulate_quantum_channel_noise(alice_circuit)
        
        # Step 3: Bob measures
        complete_circuit = self.create_bob_measurement_circuit(noisy_circuit, bob_basis)
        
        # Step 4: Execute circuit
        result = self.execute_circuit(complete_circuit)
        
        bob_measurement = result.measurements[0] if result.measurements else 0
        
        return bob_measurement, result
    
    def run_enhanced_bb84_protocol(self) -> QKDResults:
        """
        Run BB84 protocol using Qiskit quantum circuits.
        
        Returns:
            Enhanced QKD results with quantum circuit data
        """
        print("Running enhanced BB84 protocol with quantum circuits...")
        
        alice_bits = []
        alice_bases = []
        bob_bases = []
        bob_measurements = []
        circuit_results = []
        
        # Generate random bits and bases for Alice and Bob
        for i in range(self.params.num_qubits):
            # Alice's random choices
            alice_bit = random.randint(0, 1)
            alice_basis = random.choice([Basis.Z, Basis.X])
            
            # Bob's random basis choice
            bob_basis = random.choice([Basis.Z, Basis.X])
            
            # Simulate quantum exchange
            bob_measurement, circuit_result = self.simulate_bb84_qubit_exchange(
                alice_bit, alice_basis, bob_basis
            )
            
            # Store results
            alice_bits.append(alice_bit)
            alice_bases.append(alice_basis)
            bob_bases.append(bob_basis)
            bob_measurements.append(bob_measurement)
            circuit_results.append(circuit_result)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{self.params.num_qubits} qubits")
        
        # Classical sifting - keep only matching bases
        alice_sifted = []
        bob_sifted = []
        
        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:
                alice_sifted.append(alice_bits[i])
                bob_sifted.append(bob_measurements[i])
        
        # Calculate error rate
        if len(alice_sifted) > 0:
            errors = sum(1 for a, b in zip(alice_sifted, bob_sifted) if a != b)
            qber = errors / len(alice_sifted)
        else:
            qber = 0.0
        
        # Security analysis
        qber_threshold = 0.11
        eavesdropping_detected = qber > qber_threshold
        security_parameter = max(0, 1 - qber / qber_threshold)
        
        # Calculate final key length (simplified privacy amplification)
        final_key_length = int(len(alice_sifted) * security_parameter * 0.5)
        key_generation_rate = final_key_length / self.params.num_qubits
        
        print(f"✓ Quantum BB84 protocol completed:")
        print(f"  Raw key length: {len(alice_sifted)}")
        print(f"  QBER: {qber:.4f}")
        print(f"  Security parameter: {security_parameter:.4f}")
        print(f"  Final key length: {final_key_length}")
        
        return QKDResults(
            raw_key_length=len(alice_sifted),
            sifted_key_length=len(alice_sifted),
            final_key_length=final_key_length,
            error_rate=qber,
            key_generation_rate=key_generation_rate,
            security_parameter=security_parameter,
            eavesdropping_detected=eavesdropping_detected
        )
    
    def _fallback_circuit(self, bit: int, basis: Basis) -> Any:
        """Fallback circuit representation when Qiskit unavailable"""
        return {
            'type': 'fallback_circuit',
            'bit': bit,
            'basis': basis.name,
            'gates': []
        }
    
    def _fallback_execution(self, circuit: Any, shots: int) -> QuantumCircuitResult:
        """Fallback execution when Qiskit unavailable"""
        # Classical probability simulation
        measurement = random.randint(0, 1)
        probabilities = {'0': 0.6, '1': 0.4} if measurement == 0 else {'0': 0.4, '1': 0.6}
        
        return QuantumCircuitResult(
            measurements=[measurement],
            circuit=circuit,
            probabilities=probabilities,
            fidelity=0.8
        )

def create_quantum_education_circuits():
    """
    Create educational quantum circuits for visualization and learning.
    
    Returns:
        Dictionary of educational circuits
    """
    circuits = {}
    
    if not QISKIT_AVAILABLE:
        print("Qiskit not available - skipping circuit creation")
        return circuits
    
    # 1. Basic qubit states
    qc_zero = QuantumCircuit(1, 1)
    qc_zero.measure(0, 0)
    circuits['zero_state'] = qc_zero
    
    qc_one = QuantumCircuit(1, 1)
    qc_one.x(0)
    qc_one.measure(0, 0)
    circuits['one_state'] = qc_one
    
    # 2. Superposition states
    qc_plus = QuantumCircuit(1, 1)
    qc_plus.h(0)
    qc_plus.measure(0, 0)
    circuits['plus_state'] = qc_plus
    
    qc_minus = QuantumCircuit(1, 1)
    qc_minus.x(0)
    qc_minus.h(0)
    qc_minus.measure(0, 0)
    circuits['minus_state'] = qc_minus
    
    # 3. BB84 preparation examples
    bb84_prep = QuantumCircuit(2, 2)
    bb84_prep.h(0)  # |+> state
    bb84_prep.x(1)  # |1> state
    bb84_prep.barrier()
    bb84_prep.measure_all()
    circuits['bb84_preparation'] = bb84_prep
    
    print(f"✓ Created {len(circuits)} educational quantum circuits")
    
    return circuits

def demonstrate_qiskit_integration():
    """
    Demonstrate Qiskit integration with the quantum encryption system.
    """
    print("Qiskit Integration Demonstration")
    print("=" * 40)
    
    # Test parameters
    params = QKDParameters(
        num_qubits=100,  # Smaller test
        channel_noise=0.02,
        detector_efficiency=0.9,
        eavesdropping_probability=0.0
    )
    
    # Create and run Qiskit simulator
    qiskit_simulator = QiskitBB84Simulator(params)
    
    # Run enhanced BB84 protocol
    results = qiskit_simulator.run_enhanced_bb84_protocol()
    
    print("\nResults:")
    print(f"  QBER: {results.error_rate:.4f}")
    print(f"  Key Generation Rate: {results.key_generation_rate:.4f}")
    print(f"  Security Parameter: {results.security_parameter:.4f}")
    print(f"  Eavesdropping Detected: {results.eavesdropping_detected}")
    
    # Create educational circuits
    educational_circuits = create_quantum_education_circuits()
    
    print(f"\n✓ Qiskit integration demonstration completed")
    print(f"✓ Educational circuits created: {len(educational_circuits)}")
    
    return results, educational_circuits

if __name__ == "__main__":
    demonstrate_qiskit_integration()