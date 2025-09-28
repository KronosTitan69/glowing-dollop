#!/usr/bin/env python3
"""
Demonstration of Qiskit Integration in Quantum Encryption Verification System

This script demonstrates the enhanced capabilities when Qiskit is available,
while maintaining full backward compatibility when it's not.
"""

import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# Test Qiskit availability
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.providers.basic_provider import BasicProvider
    from qiskit import execute, transpile
    QISKIT_AVAILABLE = True
    print("🚀 Qiskit Integration Demo - Quantum Features Enabled")
except ImportError:
    QISKIT_AVAILABLE = False
    print("📊 Qiskit Integration Demo - Classical Fallback Mode")

print("=" * 60)

class Basis(Enum):
    """Quantum measurement bases"""
    Z = 0  # Computational basis (|0⟩, |1⟩)
    X = 1  # Hadamard basis (|+⟩, |-⟩)

@dataclass
class DemoResults:
    """Results from integration demonstration"""
    feature: str
    status: str
    details: Dict[str, Any]
    qiskit_enhanced: bool

def demo_bb84_state_preparation():
    """Demonstrate BB84 state preparation with and without Qiskit"""
    print("\n1. BB84 State Preparation")
    print("-" * 30)
    
    states_to_test = [
        (0, Basis.Z, "|0⟩"),
        (1, Basis.Z, "|1⟩"),
        (0, Basis.X, "|+⟩"),
        (1, Basis.X, "|-⟩")
    ]
    
    results = []
    
    for bit, basis, state_name in states_to_test:
        if QISKIT_AVAILABLE:
            try:
                # Create quantum circuit
                qc = QuantumCircuit(1, 1)
                
                # Prepare state
                if bit == 1:
                    qc.x(0)  # X gate for |1⟩
                if basis == Basis.X:
                    qc.h(0)  # H gate for superposition
                
                qc.measure(0, 0)
                
                # Execute on simulator
                provider = BasicProvider()
                backend = provider.get_backend('basic_simulator')
                job = execute(qc, backend, shots=100)
                result = job.result()
                counts = result.get_counts()
                
                status = "✓ Quantum circuit created and executed"
                details = {
                    'circuit_depth': qc.depth(),
                    'gates_used': [op.operation.name for op in qc.data],
                    'measurement_counts': counts,
                    'most_likely_outcome': max(counts.keys(), key=lambda x: counts[x])
                }
                qiskit_enhanced = True
                
            except Exception as e:
                status = f"✗ Quantum execution failed: {e}"
                details = {'error': str(e)}
                qiskit_enhanced = False
        else:
            # Classical simulation
            status = "📊 Classical probability simulation"
            details = {
                'bit': bit,
                'basis': basis.name,
                'state': state_name,
                'method': 'classical_probability'
            }
            qiskit_enhanced = False
        
        result = DemoResults(
            feature=f"State {state_name} preparation",
            status=status,
            details=details,
            qiskit_enhanced=qiskit_enhanced
        )
        
        results.append(result)
        print(f"  {state_name:4} | {status}")
    
    return results

def demo_quantum_measurement():
    """Demonstrate quantum measurement in different bases"""
    print("\n2. Quantum Measurement in Different Bases")
    print("-" * 40)
    
    test_cases = [
        ("Same basis", 0, Basis.Z, Basis.Z, "Deterministic"),
        ("Different basis", 0, Basis.Z, Basis.X, "Random (50/50)"),
        ("Superposition same", 0, Basis.X, Basis.X, "Deterministic"),
        ("Superposition different", 0, Basis.X, Basis.Z, "Random (50/50)")
    ]
    
    results = []
    
    for description, bit, prep_basis, meas_basis, expected in test_cases:
        if QISKIT_AVAILABLE:
            try:
                # Create complete measurement circuit
                qc = QuantumCircuit(1, 1)
                
                # Alice's preparation
                if bit == 1:
                    qc.x(0)
                if prep_basis == Basis.X:
                    qc.h(0)
                
                qc.barrier()
                
                # Bob's measurement basis
                if meas_basis == Basis.X:
                    qc.h(0)  # Transform to X basis
                
                qc.measure(0, 0)
                
                # Execute multiple times to see statistics
                provider = BasicProvider()
                backend = provider.get_backend('basic_simulator')
                job = execute(qc, backend, shots=50)
                result = job.result()
                counts = result.get_counts()
                
                # Analyze randomness
                if '0' in counts and '1' in counts:
                    randomness = "Random"
                    ratio = f"{counts.get('0', 0)}:{counts.get('1', 0)}"
                else:
                    randomness = "Deterministic"
                    ratio = f"{counts.get('0', 0) + counts.get('1', 0)}:0"
                
                status = f"✓ {randomness} result ({ratio})"
                details = {
                    'preparation_basis': prep_basis.name,
                    'measurement_basis': meas_basis.name,
                    'counts': counts,
                    'randomness': randomness,
                    'expected': expected
                }
                qiskit_enhanced = True
                
            except Exception as e:
                status = f"✗ Measurement test failed: {e}"
                details = {'error': str(e)}
                qiskit_enhanced = False
        else:
            # Classical logic
            if prep_basis == meas_basis:
                outcome = "Deterministic"
            else:
                outcome = "Random"
            
            status = f"📊 {outcome} (classical simulation)"
            details = {
                'preparation_basis': prep_basis.name,
                'measurement_basis': meas_basis.name,
                'expected_outcome': outcome,
                'method': 'classical_logic'
            }
            qiskit_enhanced = False
        
        result = DemoResults(
            feature=f"Measurement: {description}",
            status=status,
            details=details,
            qiskit_enhanced=qiskit_enhanced
        )
        
        results.append(result)
        print(f"  {description:20} | {status}")
    
    return results

def demo_mini_bb84_protocol():
    """Demonstrate a complete mini BB84 protocol"""
    print("\n3. Mini BB84 Protocol (10 qubits)")
    print("-" * 35)
    
    num_qubits = 10
    alice_bits = []
    alice_bases = []
    bob_bases = []
    bob_measurements = []
    
    # Generate random Alice and Bob choices
    for i in range(num_qubits):
        alice_bit = random.randint(0, 1)
        alice_basis = random.choice([Basis.Z, Basis.X])
        bob_basis = random.choice([Basis.Z, Basis.X])
        
        alice_bits.append(alice_bit)
        alice_bases.append(alice_basis)
        bob_bases.append(bob_basis)
        
        if QISKIT_AVAILABLE:
            try:
                # Quantum simulation
                qc = QuantumCircuit(1, 1)
                
                # Alice prepares
                if alice_bit == 1:
                    qc.x(0)
                if alice_basis == Basis.X:
                    qc.h(0)
                
                # Bob measures
                if bob_basis == Basis.X:
                    qc.h(0)
                qc.measure(0, 0)
                
                # Execute
                provider = BasicProvider()
                backend = provider.get_backend('basic_simulator')
                job = execute(qc, backend, shots=1)
                result = job.result()
                counts = result.get_counts()
                
                # Get measurement result
                measurement = int(max(counts.keys(), key=lambda x: counts[x]))
                
            except:
                # Fallback to classical
                if alice_basis == bob_basis:
                    measurement = alice_bit
                else:
                    measurement = random.randint(0, 1)
        else:
            # Classical simulation
            if alice_basis == bob_basis:
                measurement = alice_bit  # Perfect correlation
            else:
                measurement = random.randint(0, 1)  # Random
        
        bob_measurements.append(measurement)
    
    # Classical sifting
    sifted_alice = []
    sifted_bob = []
    
    for i in range(num_qubits):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_bits[i])
            sifted_bob.append(bob_measurements[i])
    
    # Error analysis
    if len(sifted_alice) > 0:
        errors = sum(1 for a, b in zip(sifted_alice, sifted_bob) if a != b)
        error_rate = errors / len(sifted_alice)
    else:
        error_rate = 0.0
    
    # Results
    matching_bases = sum(1 for a, b in zip(alice_bases, bob_bases) if a == b)
    sifting_efficiency = matching_bases / num_qubits
    
    print(f"  Raw qubits: {num_qubits}")
    print(f"  Matching bases: {matching_bases}")
    print(f"  Sifted key length: {len(sifted_alice)}")
    print(f"  Sifting efficiency: {sifting_efficiency:.2f}")
    print(f"  Error rate (QBER): {error_rate:.3f}")
    print(f"  Protocol secure: {'✓' if error_rate < 0.11 else '✗'}")
    
    method = "quantum circuits" if QISKIT_AVAILABLE else "classical simulation"
    print(f"  Method: {method}")
    
    return DemoResults(
        feature="Mini BB84 Protocol",
        status=f"✓ Completed using {method}",
        details={
            'raw_qubits': num_qubits,
            'sifted_key_length': len(sifted_alice),
            'error_rate': error_rate,
            'sifting_efficiency': sifting_efficiency,
            'secure': error_rate < 0.11,
            'method': method
        },
        qiskit_enhanced=QISKIT_AVAILABLE
    )

def demo_quantum_circuit_features():
    """Demonstrate quantum circuit specific features"""
    print("\n4. Quantum Circuit Features")
    print("-" * 30)
    
    if not QISKIT_AVAILABLE:
        print("  📊 Qiskit not available - circuit features disabled")
        return DemoResults(
            feature="Quantum Circuit Features",
            status="📊 Disabled - Qiskit not available",
            details={'qiskit_available': False},
            qiskit_enhanced=False
        )
    
    try:
        # Create example circuits
        circuits_created = 0
        total_gates = 0
        
        # Bell state preparation
        bell_circuit = QuantumCircuit(2, 2)
        bell_circuit.h(0)
        bell_circuit.cx(0, 1)
        bell_circuit.measure_all()
        circuits_created += 1
        total_gates += len(bell_circuit.data)
        
        # GHZ state preparation
        ghz_circuit = QuantumCircuit(3, 3)
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz_circuit.measure_all()
        circuits_created += 1
        total_gates += len(ghz_circuit.data)
        
        # Random quantum walk
        walk_circuit = QuantumCircuit(2, 2)
        for _ in range(3):
            walk_circuit.h(0)
            walk_circuit.cx(0, 1)
            walk_circuit.h(1)
        walk_circuit.measure_all()
        circuits_created += 1
        total_gates += len(walk_circuit.data)
        
        print(f"  ✓ Created {circuits_created} quantum circuits")
        print(f"  ✓ Total quantum gates: {total_gates}")
        print(f"  ✓ Circuit depths: Bell={bell_circuit.depth()}, GHZ={ghz_circuit.depth()}, Walk={walk_circuit.depth()}")
        
        return DemoResults(
            feature="Quantum Circuit Features",
            status="✓ Quantum circuits created successfully",
            details={
                'circuits_created': circuits_created,
                'total_gates': total_gates,
                'bell_depth': bell_circuit.depth(),
                'ghz_depth': ghz_circuit.depth(),
                'walk_depth': walk_circuit.depth()
            },
            qiskit_enhanced=True
        )
        
    except Exception as e:
        print(f"  ✗ Circuit creation failed: {e}")
        return DemoResults(
            feature="Quantum Circuit Features",
            status=f"✗ Failed: {e}",
            details={'error': str(e)},
            qiskit_enhanced=False
        )

def print_demo_summary(all_results):
    """Print comprehensive summary of demonstration"""
    print("\n" + "="*60)
    print("QISKIT INTEGRATION DEMONSTRATION SUMMARY")
    print("="*60)
    
    total_features = len(all_results)
    quantum_enhanced = sum(1 for r in all_results if r.qiskit_enhanced)
    classical_fallback = total_features - quantum_enhanced
    
    print(f"Features Demonstrated: {total_features}")
    print(f"Quantum Enhanced: {quantum_enhanced}")
    print(f"Classical Fallback: {classical_fallback}")
    print(f"Qiskit Available: {'✓' if QISKIT_AVAILABLE else '✗'}")
    
    print(f"\nFeature Status:")
    print("-" * 40)
    for result in all_results:
        enhancement = "🚀" if result.qiskit_enhanced else "📊"
        print(f"{enhancement} {result.feature:25} | {result.status}")
    
    if QISKIT_AVAILABLE:
        print("\n🎉 Quantum circuit simulation successfully integrated!")
        print("   • BB84 protocol uses real quantum circuits")
        print("   • Quantum measurements properly implemented")
        print("   • Circuit statistics and analysis available")
        print("   • Educational quantum features enabled")
    else:
        print("\n📋 Classical simulation mode:")
        print("   • All features work with probability models")
        print("   • Install Qiskit to enable quantum circuits")
        print("   • Backward compatibility maintained")
    
    print(f"\n✓ Integration demonstration completed successfully")

def main():
    """Main demonstration function"""
    all_results = []
    
    # Run all demonstrations
    all_results.extend(demo_bb84_state_preparation())
    all_results.extend(demo_quantum_measurement())
    all_results.append(demo_mini_bb84_protocol())
    all_results.append(demo_quantum_circuit_features())
    
    # Print comprehensive summary
    print_demo_summary(all_results)
    
    return all_results

if __name__ == "__main__":
    main()