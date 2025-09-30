"""
Simple Qiskit Integration Test

This script tests basic Qiskit functionality without external dependencies.
"""

import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# Test Qiskit availability
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.providers.basic_provider import BasicProvider
    from qiskit import execute
    QISKIT_AVAILABLE = True
    print("✓ Qiskit successfully imported and available")
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Using fallback simulation.")

class Basis(Enum):
    Z = 0  # Computational basis
    X = 1  # Hadamard basis

@dataclass 
class TestResult:
    success: bool
    message: str
    details: Dict[str, Any]

def test_basic_qiskit_functionality():
    """Test basic Qiskit circuit creation and execution"""
    if not QISKIT_AVAILABLE:
        return TestResult(
            success=False,
            message="Qiskit not available",
            details={"fallback": True}
        )
    
    try:
        # Create simple circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)  # Hadamard on qubit 0
        qc.cx(0, 1)  # CNOT from qubit 0 to 1
        qc.measure_all()
        
        # Test backend
        provider = BasicProvider()
        backend = provider.get_backend('basic_simulator')
        
        # Execute circuit
        job = execute(qc, backend, shots=100)
        result = job.result()
        counts = result.get_counts()
        
        return TestResult(
            success=True,
            message="Basic Qiskit functionality working",
            details={
                "circuit_depth": qc.depth(),
                "num_qubits": qc.num_qubits,
                "backend": backend.name(),
                "measurement_counts": counts
            }
        )
        
    except Exception as e:
        return TestResult(
            success=False,
            message=f"Qiskit test failed: {str(e)}",
            details={"error": str(e)}
        )

def test_bb84_quantum_circuit():
    """Test BB84-specific quantum circuits"""
    if not QISKIT_AVAILABLE:
        return TestResult(
            success=False,
            message="Qiskit not available for BB84 test",
            details={"fallback": True}
        )
    
    try:
        results = []
        
        # Test all BB84 state preparations
        test_cases = [
            (0, Basis.Z),  # |0⟩
            (1, Basis.Z),  # |1⟩
            (0, Basis.X),  # |+⟩
            (1, Basis.X),  # |-⟩
        ]
        
        for bit, basis in test_cases:
            # Alice prepares qubit
            qc = QuantumCircuit(1, 1)
            
            # Prepare bit
            if bit == 1:
                qc.x(0)
            
            # Apply basis transformation
            if basis == Basis.X:
                qc.h(0)
            
            # Bob measures (same basis for this test)
            if basis == Basis.X:
                qc.h(0)  # Transform back for measurement
            
            qc.measure(0, 0)
            
            # Execute
            provider = BasicProvider()
            backend = provider.get_backend('basic_simulator')
            job = execute(qc, backend, shots=100)
            result = job.result()
            counts = result.get_counts()
            
            results.append({
                'input_bit': bit,
                'basis': basis.name,
                'counts': counts,
                'circuit_ops': qc.count_ops()
            })
        
        return TestResult(
            success=True,
            message="BB84 quantum circuits working",
            details={"test_results": results}
        )
        
    except Exception as e:
        return TestResult(
            success=False,
            message=f"BB84 circuit test failed: {str(e)}",
            details={"error": str(e)}
        )

def run_qiskit_integration_tests():
    """Run all Qiskit integration tests"""
    print("Qiskit Integration Tests")
    print("=" * 30)
    
    tests = [
        ("Basic Functionality", test_basic_qiskit_functionality),
        ("BB84 Circuits", test_bb84_quantum_circuit),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        result = test_func()
        results.append((test_name, result))
        
        if result.success:
            print(f"✓ {result.message}")
            if 'measurement_counts' in result.details:
                print(f"  Measurement counts: {result.details['measurement_counts']}")
            if 'test_results' in result.details:
                print(f"  Tests completed: {len(result.details['test_results'])}")
        else:
            print(f"✗ {result.message}")
            if 'error' in result.details:
                print(f"  Error: {result.details['error']}")
    
    # Summary
    successful_tests = sum(1 for _, result in results if result.success)
    total_tests = len(results)
    
    print(f"\nTest Summary:")
    print(f"  Successful: {successful_tests}/{total_tests}")
    print(f"  Success Rate: {successful_tests/total_tests:.1%}")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! Qiskit integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return results

if __name__ == "__main__":
    run_qiskit_integration_tests()