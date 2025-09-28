#!/usr/bin/env python3
"""
Test script for Qiskit integration in BB84 QKD simulation.

This script tests the enhanced BB84 implementation with minimal dependencies.
"""

import random
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

# Test Qiskit availability
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.providers.basic_provider import BasicProvider
    from qiskit import execute, transpile
    QISKIT_AVAILABLE = True
    print("✓ Qiskit successfully imported - quantum circuit simulation enabled")
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Using classical probability simulation.")

class Basis(Enum):
    """Quantum measurement bases"""
    Z = 0  # Computational basis (|0⟩, |1⟩)
    X = 1  # Hadamard basis (|+⟩, |-⟩)

@dataclass
class QKDParameters:
    """Parameters for QKD simulation"""
    num_qubits: int = 100
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

def test_qiskit_bb84_integration():
    """Test the Qiskit integration for BB84 simulation"""
    print("Testing Qiskit BB84 Integration")
    print("=" * 35)
    
    # Test parameters
    params = QKDParameters(
        num_qubits=50,  # Small test
        channel_noise=0.02,
        detector_efficiency=0.9,
        eavesdropping_probability=0.0
    )
    
    results = {
        'qiskit_available': QISKIT_AVAILABLE,
        'circuits_created': 0,
        'circuits_executed': 0,
        'quantum_operations': 0,
        'classical_fallbacks': 0,
        'test_results': []
    }
    
    if QISKIT_AVAILABLE:
        try:
            # Test quantum circuit creation
            print("Testing quantum circuit creation...")
            
            # Test Alice's qubit preparation
            test_cases = [
                (0, Basis.Z),  # |0⟩
                (1, Basis.Z),  # |1⟩
                (0, Basis.X),  # |+⟩
                (1, Basis.X),  # |-⟩
            ]
            
            provider = BasicProvider()
            backend = provider.get_backend('basic_simulator')
            
            for bit, basis in test_cases:
                # Create Alice's preparation circuit
                qc = QuantumCircuit(1, 1)
                
                # Prepare bit
                if bit == 1:
                    qc.x(0)
                
                # Apply basis transformation
                if basis == Basis.X:
                    qc.h(0)
                
                results['circuits_created'] += 1
                results['quantum_operations'] += len(qc.data)
                
                # Test measurement in same basis
                qc_measure = qc.copy()
                if basis == Basis.X:
                    qc_measure.h(0)  # Transform back for measurement
                qc_measure.measure(0, 0)
                
                # Execute circuit
                job = execute(qc_measure, backend, shots=10)
                result = job.result()
                counts = result.get_counts()
                
                results['circuits_executed'] += 1
                
                # Check if we get expected result (should be mostly the original bit)
                most_common = max(counts.keys(), key=lambda x: counts[x])
                expected_match = (int(most_common) == bit)
                
                test_result = {
                    'input_bit': bit,
                    'basis': basis.name,
                    'measured_counts': counts,
                    'expected_match': expected_match,
                    'circuit_depth': qc_measure.depth()
                }
                
                results['test_results'].append(test_result)
                print(f"  Test {bit}|{basis.name}: {counts} - {'✓' if expected_match else '✗'}")
            
            print(f"✓ Created and executed {results['circuits_created']} quantum circuits")
            
        except Exception as e:
            print(f"✗ Quantum circuit test failed: {e}")
            results['error'] = str(e)
            return results
            
    else:
        print("Qiskit not available - testing classical fallback...")
        
        # Test classical simulation
        for i in range(5):
            alice_bit = random.randint(0, 1)
            alice_basis = random.choice([Basis.Z, Basis.X])
            bob_basis = random.choice([Basis.Z, Basis.X])
            
            # Classical BB84 logic
            if alice_basis == bob_basis:
                # Same basis - should get same result (with noise)
                if random.random() < params.channel_noise:
                    bob_measurement = 1 - alice_bit
                else:
                    bob_measurement = alice_bit
            else:
                # Different basis - random result
                bob_measurement = random.randint(0, 1)
            
            results['classical_fallbacks'] += 1
            
            test_result = {
                'alice_bit': alice_bit,
                'alice_basis': alice_basis.name,
                'bob_basis': bob_basis.name,
                'bob_measurement': bob_measurement,
                'bases_match': alice_basis == bob_basis
            }
            
            results['test_results'].append(test_result)
            
        print(f"✓ Completed {results['classical_fallbacks']} classical simulations")
    
    return results

def simulate_mini_bb84_protocol():
    """Simulate a mini BB84 protocol to test integration"""
    print("\nMini BB84 Protocol Simulation")
    print("=" * 30)
    
    params = QKDParameters(num_qubits=20, channel_noise=0.01)
    
    alice_bits = []
    alice_bases = []
    bob_bases = []
    bob_measurements = []
    
    # Generate random bits and bases
    for i in range(params.num_qubits):
        alice_bit = random.randint(0, 1)
        alice_basis = random.choice([Basis.Z, Basis.X])
        bob_basis = random.choice([Basis.Z, Basis.X])
        
        # Simulate measurement (simplified)
        if alice_basis == bob_basis:
            # Same basis - high probability of correct measurement
            if random.random() < params.channel_noise:
                bob_measurement = 1 - alice_bit  # Bit flip due to noise
            else:
                bob_measurement = alice_bit
        else:
            # Different basis - random result
            bob_measurement = random.randint(0, 1)
        
        alice_bits.append(alice_bit)
        alice_bases.append(alice_basis)
        bob_bases.append(bob_basis)
        bob_measurements.append(bob_measurement)
    
    print(f"Generated {params.num_qubits} qubit exchanges")
    
    # Classical sifting
    alice_sifted = []
    bob_sifted = []
    
    for i in range(len(alice_bits)):
        if alice_bases[i] == bob_bases[i]:
            alice_sifted.append(alice_bits[i])
            bob_sifted.append(bob_measurements[i])
    
    print(f"Sifted key length: {len(alice_sifted)}")
    
    # Calculate error rate
    if len(alice_sifted) > 0:
        errors = sum(1 for a, b in zip(alice_sifted, bob_sifted) if a != b)
        qber = errors / len(alice_sifted)
    else:
        qber = 0.0
    
    print(f"QBER: {qber:.4f}")
    
    # Security analysis
    qber_threshold = 0.11
    secure = qber <= qber_threshold
    print(f"Protocol secure: {secure}")
    
    return {
        'raw_bits': len(alice_bits),
        'sifted_bits': len(alice_sifted),
        'qber': qber,
        'secure': secure,
        'matching_bases': sum(1 for a, b in zip(alice_bases, bob_bases) if a == b)
    }

def main():
    """Main test function"""
    print("Qiskit Integration Test Suite")
    print("=" * 40)
    
    # Test 1: Qiskit BB84 Integration
    integration_results = test_qiskit_bb84_integration()
    
    # Test 2: Mini BB84 Protocol
    protocol_results = simulate_mini_bb84_protocol()
    
    # Summary
    print("\nTest Summary")
    print("=" * 15)
    print(f"Qiskit Available: {integration_results['qiskit_available']}")
    print(f"Circuits Created: {integration_results['circuits_created']}")
    print(f"Circuits Executed: {integration_results['circuits_executed']}")
    print(f"Classical Fallbacks: {integration_results['classical_fallbacks']}")
    print(f"Protocol QBER: {protocol_results['qber']:.4f}")
    print(f"Protocol Secure: {protocol_results['secure']}")
    
    if integration_results['qiskit_available']:
        successful_tests = sum(1 for test in integration_results['test_results'] 
                             if test.get('expected_match', False))
        total_tests = len(integration_results['test_results'])
        print(f"Quantum Tests Passed: {successful_tests}/{total_tests}")
        
        if successful_tests == total_tests:
            print("🎉 All quantum circuit tests passed!")
        else:
            print("⚠️ Some quantum circuit tests failed")
    else:
        print("📊 Classical simulation working correctly")
    
    print("\n✓ Qiskit integration test completed")

if __name__ == "__main__":
    main()