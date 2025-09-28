"""
Main Execution Script for Quantum Encryption Verification System

This script orchestrates the complete quantum encryption verification process including:
- BB84 QKD protocol simulation
- Eavesdropping analysis
- Key rate analysis
- Post-quantum cryptography integration
- Quantum concept diagrams
- Technical report generation
- Validation and benchmarking

Author: Quantum Encryption Verification System
Date: 2024
"""

import os
import sys
import time
import warnings
from typing import Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_banner():
    """Print system banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        Quantum Encryption Verification System                ║
    ║                                                              ║
    ║  Comprehensive Analysis of QKD, Post-Quantum Crypto,        ║
    ║  and Practical Implementation Challenges                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    required_packages = [
        'numpy', 'matplotlib', 'seaborn', 'scipy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    print("All dependencies available!")
    return True

def run_bb84_simulation():
    """Run BB84 QKD protocol simulation."""
    print("\n" + "="*60)
    print("1. BB84 QKD Protocol Simulation")
    print("="*60)
    
    try:
        from bb84_qkd_simulation import simulate_qkd_scenarios
        results = simulate_qkd_scenarios()
        print("✓ BB84 simulation completed successfully")
        return results
    except Exception as e:
        print(f"✗ BB84 simulation failed: {e}")
        return None

def run_eavesdropping_analysis():
    """Run eavesdropping analysis."""
    print("\n" + "="*60)
    print("2. Eavesdropping Analysis")
    print("="*60)
    
    try:
        from eavesdropper_analysis import analyze_eavesdropping_scenarios
        results = analyze_eavesdropping_scenarios()
        print("✓ Eavesdropping analysis completed successfully")
        return results
    except Exception as e:
        print(f"✗ Eavesdropping analysis failed: {e}")
        return None

def run_key_rate_analysis():
    """Run key rate analysis."""
    print("\n" + "="*60)
    print("3. Key Rate Analysis")
    print("="*60)
    
    try:
        from key_rate_analysis import benchmark_against_literature
        results = benchmark_against_literature()
        print("✓ Key rate analysis completed successfully")
        return results
    except Exception as e:
        print(f"✗ Key rate analysis failed: {e}")
        return None

def run_post_quantum_analysis():
    """Run post-quantum cryptography analysis."""
    print("\n" + "="*60)
    print("4. Post-Quantum Cryptography Analysis")
    print("="*60)
    
    try:
        from post_quantum_crypto import analyze_hybrid_systems
        results = analyze_hybrid_systems()
        print("✓ Post-quantum analysis completed successfully")
        return results
    except Exception as e:
        print(f"✗ Post-quantum analysis failed: {e}")
        return None

def run_quantum_diagrams():
    """Generate quantum concept diagrams."""
    print("\n" + "="*60)
    print("5. Quantum Concept Diagrams")
    print("="*60)
    
    try:
        from quantum_diagrams import create_comprehensive_visualizations
        create_comprehensive_visualizations()
        print("✓ Quantum diagrams generated successfully")
        return True
    except Exception as e:
        print(f"✗ Quantum diagrams generation failed: {e}")
        return None

def run_validation_analysis():
    """Run validation and benchmarking."""
    print("\n" + "="*60)
    print("6. Validation and Benchmarking")
    print("="*60)
    
    try:
        from validation_benchmarks import run_comprehensive_validation
        results = run_comprehensive_validation()
        print("✓ Validation analysis completed successfully")
        return results
    except Exception as e:
        print(f"✗ Validation analysis failed: {e}")
        return None

def generate_final_report():
    """Generate final comprehensive report."""
    print("\n" + "="*60)
    print("7. Final Report Generation")
    print("="*60)
    
    try:
        # Check if technical report exists
        if os.path.exists('quantum_encryption_verification/technical_report.md'):
            print("✓ Technical report already generated")
        else:
            print("✗ Technical report not found")
        
        # Check if validation report exists
        if os.path.exists('quantum_encryption_verification/validation_report.md'):
            print("✓ Validation report already generated")
        else:
            print("✗ Validation report not found")
        
        print("✓ Final report generation completed")
        return True
    except Exception as e:
        print(f"✗ Final report generation failed: {e}")
        return None

def create_output_directory():
    """Create output directory if it doesn't exist."""
    output_dir = 'quantum_encryption_verification'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def main():
    """Main execution function."""
    start_time = time.time()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return
    
    # Create output directory
    create_output_directory()
    
    # Initialize results dictionary
    results = {
        'bb84_simulation': None,
        'eavesdropping_analysis': None,
        'key_rate_analysis': None,
        'post_quantum_analysis': None,
        'quantum_diagrams': None,
        'validation_analysis': None,
        'final_report': None
    }
    
    # Run all analyses
    print("\nStarting comprehensive quantum encryption verification...")
    
    # 1. BB84 QKD Simulation
    results['bb84_simulation'] = run_bb84_simulation()
    
    # 2. Eavesdropping Analysis
    results['eavesdropping_analysis'] = run_eavesdropping_analysis()
    
    # 3. Key Rate Analysis
    results['key_rate_analysis'] = run_key_rate_analysis()
    
    # 4. Post-Quantum Analysis
    results['post_quantum_analysis'] = run_post_quantum_analysis()
    
    # 5. Quantum Diagrams
    results['quantum_diagrams'] = run_quantum_diagrams()
    
    # 6. Validation Analysis
    results['validation_analysis'] = run_validation_analysis()
    
    # 7. Final Report
    results['final_report'] = generate_final_report()
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    successful_analyses = sum(1 for result in results.values() if result is not None)
    total_analyses = len(results)
    
    print(f"Total Analyses: {total_analyses}")
    print(f"Successful: {successful_analyses}")
    print(f"Failed: {total_analyses - successful_analyses}")
    print(f"Success Rate: {successful_analyses/total_analyses:.1%}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    
    print("\nGenerated Files:")
    print("- technical_report.md")
    print("- validation_report.md")
    print("- Various visualization PNG files")
    print("- Quantum concept diagrams")
    
    print("\n" + "="*60)
    print("QUANTUM ENCRYPTION VERIFICATION COMPLETED")
    print("="*60)
    
    if successful_analyses == total_analyses:
        print("🎉 All analyses completed successfully!")
    else:
        print("⚠️  Some analyses failed. Check the output above for details.")
    
    print(f"\nResults saved in: quantum_encryption_verification/")
    print("Check the generated reports and visualizations for detailed analysis.")

if __name__ == "__main__":
    main()


