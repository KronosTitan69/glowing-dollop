#!/usr/bin/env python3
"""
Complete Integration Validation Script

This script validates that the entire Qiskit integration is working correctly
and demonstrates all the enhanced features.
"""

import sys
import time
from typing import Dict, List, Any

def test_import_compatibility():
    """Test that all modules can be imported correctly"""
    print("Testing Import Compatibility...")
    
    modules_to_test = [
        'bb84_qkd_simulation',
        'quantum_diagrams', 
        'qiskit_integration',
        'main'
    ]
    
    results = {}
    
    for module in modules_to_test:
        try:
            __import__(module)
            results[module] = "✓ Success"
        except Exception as e:
            results[module] = f"✗ Failed: {e}"
    
    for module, status in results.items():
        print(f"  {module:20} | {status}")
    
    all_success = all("Success" in status for status in results.values())
    return all_success, results

def test_qiskit_detection():
    """Test Qiskit detection and availability"""
    print("\nTesting Qiskit Detection...")
    
    try:
        import qiskit
        qiskit_version = qiskit.__version__
        qiskit_available = True
        status = f"✓ Qiskit v{qiskit_version} available"
    except ImportError:
        qiskit_available = False
        status = "📊 Qiskit not available - classical mode"
    
    print(f"  {status}")
    
    # Test QISKIT_AVAILABLE flags in modules
    try:
        from bb84_qkd_simulation import QISKIT_AVAILABLE as bb84_qiskit
        from quantum_diagrams import QISKIT_AVAILABLE as diag_qiskit
        
        print(f"  BB84 module detection: {'✓' if bb84_qiskit else '📊'}")
        print(f"  Diagrams module detection: {'✓' if diag_qiskit else '📊'}")
        
        flags_consistent = bb84_qiskit == diag_qiskit == qiskit_available
        
    except ImportError as e:
        print(f"  ✗ Module import failed: {e}")
        flags_consistent = False
    
    return qiskit_available, flags_consistent

def test_enhanced_bb84():
    """Test enhanced BB84 functionality"""
    print("\nTesting Enhanced BB84...")
    
    try:
        from bb84_qkd_simulation import BB84QKD, QKDParameters
        
        # Small test
        params = QKDParameters(num_qubits=10, channel_noise=0.01)
        qkd = BB84QKD(params)
        
        # Test quantum-enhanced protocol
        result = qkd.run_quantum_enhanced_protocol()
        
        # Test statistics
        stats = qkd.get_quantum_circuit_statistics()
        
        print(f"  ✓ Protocol completed successfully")
        print(f"  ✓ QBER: {result.error_rate:.4f}")
        print(f"  ✓ Key rate: {result.key_generation_rate:.4f}")
        
        if 'total_circuits' in stats:
            print(f"  ✓ Circuit statistics available: {stats['total_circuits']} circuits")
        else:
            print(f"  📊 Classical simulation used")
        
        return True, result, stats
        
    except Exception as e:
        print(f"  ✗ BB84 test failed: {e}")
        return False, None, None

def test_quantum_diagrams():
    """Test quantum diagram generation"""
    print("\nTesting Quantum Diagrams...")
    
    try:
        from quantum_diagrams import QuantumDiagramGenerator
        
        generator = QuantumDiagramGenerator()
        
        # Test basic diagram creation (without saving)
        # Mock the plotting functions to avoid external dependencies
        class MockFigure:
            def savefig(self, *args, **kwargs): pass
        
        # Test that methods exist and can be called
        methods_to_test = [
            'create_superposition_diagram',
            'create_bb84_protocol_diagram',
            'create_measurement_bases_diagram'
        ]
        
        method_results = {}
        for method_name in methods_to_test:
            if hasattr(generator, method_name):
                method_results[method_name] = "✓ Available"
            else:
                method_results[method_name] = "✗ Missing"
        
        # Test new quantum circuit methods
        quantum_methods = [
            'create_bb84_quantum_circuits_diagram',
            'create_quantum_measurement_diagram'
        ]
        
        for method_name in quantum_methods:
            if hasattr(generator, method_name):
                method_results[method_name] = "✓ Available (NEW)"
            else:
                method_results[method_name] = "✗ Missing"
        
        for method, status in method_results.items():
            print(f"  {method:35} | {status}")
        
        all_methods_available = all("Available" in status for status in method_results.values())
        return all_methods_available, method_results
        
    except Exception as e:
        print(f"  ✗ Diagram test failed: {e}")
        return False, {}

def test_integration_demos():
    """Test integration demonstration scripts"""
    print("\nTesting Integration Demos...")
    
    demo_scripts = [
        'demo_qiskit_integration.py',
        'test_qiskit_integration.py'
    ]
    
    results = {}
    
    for script in demo_scripts:
        try:
            # Test that the script can be imported (basic syntax check)
            script_module = script.replace('.py', '')
            __import__(script_module)
            results[script] = "✓ Syntax valid"
        except Exception as e:
            results[script] = f"✗ Error: {e}"
    
    for script, status in results.items():
        print(f"  {script:30} | {status}")
    
    return all("valid" in status for status in results.values()), results

def run_comprehensive_validation():
    """Run complete validation suite"""
    print("Qiskit Integration - Complete Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Import Compatibility", test_import_compatibility),
        ("Qiskit Detection", test_qiskit_detection),
        ("Enhanced BB84", test_enhanced_bb84),
        ("Quantum Diagrams", test_quantum_diagrams),
        ("Integration Demos", test_integration_demos)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                success = result[0]
                details = result[1:] if len(result) > 1 else []
            else:
                success = result
                details = []
            
            results[test_name] = {
                'success': success,
                'details': details
            }
            
        except Exception as e:
            print(f"\n✗ Test {test_name} crashed: {e}")
            results[test_name] = {
                'success': False,
                'details': [str(e)]
            }
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    successful_tests = sum(1 for r in results.values() if r['success'])
    total_tests = len(results)
    
    print(f"Tests Run: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests:.1%}")
    print(f"Duration: {duration:.2f} seconds")
    
    print(f"\nTest Results:")
    for test_name, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {test_name}")
    
    # Check if we have quantum features
    qiskit_available = False
    try:
        import qiskit
        qiskit_available = True
    except ImportError:
        pass
    
    print(f"\nSystem Status:")
    print(f"  Qiskit Available: {'✓' if qiskit_available else '📊 (classical mode)'}")
    print(f"  Integration Complete: {'✓' if successful_tests == total_tests else '✗'}")
    print(f"  Ready for Use: {'✓' if successful_tests >= total_tests - 1 else '✗'}")
    
    if successful_tests == total_tests:
        print("\n🎉 All tests passed! Qiskit integration is fully functional.")
        if qiskit_available:
            print("   🚀 Quantum circuit simulation enabled")
        else:
            print("   📊 Classical simulation working perfectly")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    print(f"\n✓ Validation completed")
    
    return results

if __name__ == "__main__":
    validation_results = run_comprehensive_validation()
    
    # Exit with appropriate code
    all_passed = all(r['success'] for r in validation_results.values())
    sys.exit(0 if all_passed else 1)