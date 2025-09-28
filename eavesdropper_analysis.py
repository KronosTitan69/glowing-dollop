"""
Eavesdropper Analysis for Quantum Key Distribution

This module implements comprehensive eavesdropping analysis including:
- Intercept-resend attacks
- Photon number splitting attacks
- Trojan horse attacks
- Error rate analysis under different attack scenarios
- Security parameter calculations

Author: Quantum Encryption Verification System
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from bb84_qkd_simulation import BB84QKD, QKDParameters, QKDResults

class AttackType(Enum):
    """Types of eavesdropping attacks"""
    INTERCEPT_RESEND = "intercept_resend"
    PHOTON_NUMBER_SPLITTING = "photon_number_splitting"
    TROJAN_HORSE = "trojan_horse"
    BEAM_SPLITTING = "beam_splitting"
    NO_ATTACK = "no_attack"

@dataclass
class AttackParameters:
    """Parameters for eavesdropping attacks"""
    attack_type: AttackType
    attack_probability: float
    photon_number_threshold: int = 2
    trojan_horse_probability: float = 0.1
    beam_splitting_ratio: float = 0.5

class Eavesdropper:
    """
    Advanced eavesdropper implementing various attack strategies.
    
    This class models sophisticated eavesdropping attacks including:
    - Intercept-resend attacks
    - Photon number splitting attacks
    - Trojan horse attacks
    - Beam splitting attacks
    """
    
    def __init__(self, attack_params: AttackParameters):
        self.attack_params = attack_params
        self.attack_success_count = 0
        self.total_attacks = 0
        
    def intercept_resend_attack(self, qubit_state: Tuple[int, int]) -> Tuple[int, int]:
        """
        Perform intercept-resend attack.
        
        Args:
            qubit_state: (bit, basis) tuple from Alice
            
        Returns:
            Modified qubit state after attack
        """
        if random.random() > self.attack_params.attack_probability:
            return qubit_state
            
        self.total_attacks += 1
        bit, basis = qubit_state
        
        # Eve randomly chooses a basis for measurement
        eve_basis = random.choice([0, 1])  # 0 = Z, 1 = X
        
        # Eve measures in her chosen basis
        if eve_basis == basis:
            # Same basis - Eve gets correct result
            eve_result = bit
            self.attack_success_count += 1
        else:
            # Different basis - Eve gets random result
            eve_result = random.randint(0, 1)
        
        # Eve resends in her chosen basis
        return (eve_result, eve_basis)
    
    def photon_number_splitting_attack(self, qubit_state: Tuple[int, int], 
                                      photon_number: int) -> Tuple[int, int]:
        """
        Perform photon number splitting attack.
        
        Args:
            qubit_state: (bit, basis) tuple from Alice
            photon_number: Number of photons in the pulse
            
        Returns:
            Modified qubit state after attack
        """
        if photon_number < self.attack_params.photon_number_threshold:
            return qubit_state
            
        if random.random() > self.attack_params.attack_probability:
            return qubit_state
            
        self.total_attacks += 1
        bit, basis = qubit_state
        
        # Eve splits off one photon and measures it
        # She resends the remaining photons
        eve_basis = random.choice([0, 1])
        
        if eve_basis == basis:
            eve_result = bit
            self.attack_success_count += 1
        else:
            eve_result = random.randint(0, 1)
        
        return (eve_result, eve_basis)
    
    def trojan_horse_attack(self, qubit_state: Tuple[int, int]) -> Tuple[int, int]:
        """
        Perform trojan horse attack.
        
        Args:
            qubit_state: (bit, basis) tuple from Alice
            
        Returns:
            Modified qubit state after attack
        """
        if random.random() > self.attack_params.trojan_horse_probability:
            return qubit_state
            
        self.total_attacks += 1
        bit, basis = qubit_state
        
        # Trojan horse attack introduces additional errors
        if random.random() < 0.1:  # 10% chance of introducing error
            bit = 1 - bit
            
        return (bit, basis)
    
    def beam_splitting_attack(self, qubit_state: Tuple[int, int]) -> Tuple[int, int]:
        """
        Perform beam splitting attack.
        
        Args:
            qubit_state: (bit, basis) tuple from Alice
            
        Returns:
            Modified qubit state after attack
        """
        if random.random() > self.attack_params.attack_probability:
            return qubit_state
            
        self.total_attacks += 1
        bit, basis = qubit_state
        
        # Eve splits the beam and measures a portion
        split_ratio = self.attack_params.beam_splitting_ratio
        if random.random() < split_ratio:
            # Eve measures this portion
            eve_basis = random.choice([0, 1])
            if eve_basis == basis:
                eve_result = bit
                self.attack_success_count += 1
            else:
                eve_result = random.randint(0, 1)
        else:
            # Eve doesn't measure this portion
            eve_result = bit
            
        return (eve_result, basis)
    
    def attack_qubit(self, qubit_state: Tuple[int, int], 
                    photon_number: int = 1) -> Tuple[int, int]:
        """
        Apply the configured attack to a qubit.
        
        Args:
            qubit_state: (bit, basis) tuple from Alice
            photon_number: Number of photons in the pulse
            
        Returns:
            Modified qubit state after attack
        """
        if self.attack_params.attack_type == AttackType.INTERCEPT_RESEND:
            return self.intercept_resend_attack(qubit_state)
        elif self.attack_params.attack_type == AttackType.PHOTON_NUMBER_SPLITTING:
            return self.photon_number_splitting_attack(qubit_state, photon_number)
        elif self.attack_params.attack_type == AttackType.TROJAN_HORSE:
            return self.trojan_horse_attack(qubit_state)
        elif self.attack_params.attack_type == AttackType.BEAM_SPLITTING:
            return self.beam_splitting_attack(qubit_state)
        else:
            return qubit_state
    
    def get_attack_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the eavesdropper's attacks.
        
        Returns:
            Dictionary with attack statistics
        """
        if self.total_attacks == 0:
            return {
                'attack_success_rate': 0.0,
                'total_attacks': 0,
                'information_gained': 0.0
            }
        
        success_rate = self.attack_success_count / self.total_attacks
        information_gained = success_rate * self.attack_params.attack_probability
        
        return {
            'attack_success_rate': success_rate,
            'total_attacks': self.total_attacks,
            'information_gained': information_gained
        }

class EnhancedBB84QKD(BB84QKD):
    """
    Enhanced BB84 QKD with advanced eavesdropping analysis.
    
    This class extends the basic BB84 implementation with:
    - Multiple attack scenarios
    - Detailed error analysis
    - Security parameter calculations
    - Attack detection mechanisms
    """
    
    def __init__(self, parameters: QKDParameters, eavesdropper: Optional[Eavesdropper] = None):
        super().__init__(parameters)
        self.eavesdropper = eavesdropper
        self.attack_statistics = {}
        
    def bob_measure_qubits_with_eavesdropper(self, alice_preparations: List[Tuple[int, int]]) -> List[int]:
        """
        Bob measures qubits with potential eavesdropping.
        
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
            
            # Apply eavesdropper attack if present
            if self.eavesdropper:
                attacked_qubit = self.eavesdropper.attack_qubit(noisy_qubit)
            else:
                attacked_qubit = noisy_qubit
            
            # Bob randomly chooses measurement basis
            bob_basis = random.choice([0, 1])  # 0 = Z, 1 = X
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
    
    def run_protocol_with_eavesdropping(self) -> QKDResults:
        """
        Run the complete BB84 protocol with eavesdropping analysis.
        
        Returns:
            QKDResults object with all simulation results
        """
        # Step 1: Alice prepares qubits
        alice_preparations = self.alice_prepare_qubits()
        
        # Step 2: Bob measures qubits (with potential eavesdropping)
        bob_measurements = self.bob_measure_qubits_with_eavesdropper(alice_preparations)
        
        # Step 3: Classical sifting
        alice_sifted, bob_sifted, sifting_efficiency = self.classical_sifting()
        
        # Step 4: Calculate error rate
        qber = self.calculate_error_rate(alice_sifted, bob_sifted)
        
        # Step 5: Security analysis
        eavesdropping_detected, security_parameter = self.security_analysis(qber)
        
        # Step 6: Privacy amplification
        final_key_length = int(len(alice_sifted) * security_parameter * 0.5)
        
        # Calculate key generation rate
        key_generation_rate = final_key_length / self.params.num_qubits
        
        # Store attack statistics
        if self.eavesdropper:
            self.attack_statistics = self.eavesdropper.get_attack_statistics()
        
        return QKDResults(
            raw_key_length=len(alice_sifted),
            sifted_key_length=len(alice_sifted),
            final_key_length=final_key_length,
            error_rate=qber,
            key_generation_rate=key_generation_rate,
            security_parameter=security_parameter,
            eavesdropping_detected=eavesdropping_detected
        )

def analyze_eavesdropping_scenarios():
    """
    Analyze different eavesdropping scenarios and their effects.
    
    Returns:
        Dictionary of results for different attack scenarios
    """
    base_params = QKDParameters(
        num_qubits=1000,
        channel_noise=0.01,
        detector_efficiency=0.8
    )
    
    attack_scenarios = {
        'no_attack': AttackParameters(
            attack_type=AttackType.NO_ATTACK,
            attack_probability=0.0
        ),
        'light_intercept_resend': AttackParameters(
            attack_type=AttackType.INTERCEPT_RESEND,
            attack_probability=0.1
        ),
        'heavy_intercept_resend': AttackParameters(
            attack_type=AttackType.INTERCEPT_RESEND,
            attack_probability=0.3
        ),
        'photon_number_splitting': AttackParameters(
            attack_type=AttackType.PHOTON_NUMBER_SPLITTING,
            attack_probability=0.2,
            photon_number_threshold=2
        ),
        'trojan_horse': AttackParameters(
            attack_type=AttackType.TROJAN_HORSE,
            attack_probability=0.15,
            trojan_horse_probability=0.1
        ),
        'beam_splitting': AttackParameters(
            attack_type=AttackType.BEAM_SPLITTING,
            attack_probability=0.25,
            beam_splitting_ratio=0.5
        )
    }
    
    results = {}
    
    for scenario_name, attack_params in attack_scenarios.items():
        print(f"Analyzing {scenario_name} scenario...")
        
        eavesdropper = Eavesdropper(attack_params)
        qkd = EnhancedBB84QKD(base_params, eavesdropper)
        result = qkd.run_protocol_with_eavesdropping()
        
        results[scenario_name] = {
            'qkd_result': result,
            'attack_stats': qkd.attack_statistics
        }
        
        print(f"  QBER: {result.error_rate:.4f}")
        print(f"  Key Generation Rate: {result.key_generation_rate:.4f}")
        print(f"  Eavesdropping Detected: {result.eavesdropping_detected}")
        if qkd.attack_statistics:
            print(f"  Attack Success Rate: {qkd.attack_statistics['attack_success_rate']:.4f}")
            print(f"  Information Gained: {qkd.attack_statistics['information_gained']:.4f}")
        print()
    
    return results

def plot_eavesdropping_analysis(results: Dict):
    """
    Create visualizations for eavesdropping analysis.
    
    Args:
        results: Dictionary of simulation results
    """
    scenarios = list(results.keys())
    qbers = [results[scenario]['qkd_result'].error_rate for scenario in scenarios]
    key_rates = [results[scenario]['qkd_result'].key_generation_rate for scenario in scenarios]
    attack_success_rates = []
    information_gained = []
    
    for scenario in scenarios:
        attack_stats = results[scenario]['attack_stats']
        if attack_stats:
            attack_success_rates.append(attack_stats['attack_success_rate'])
            information_gained.append(attack_stats['information_gained'])
        else:
            attack_success_rates.append(0.0)
            information_gained.append(0.0)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # QBER vs Attack Scenarios
    ax1.bar(scenarios, qbers, color='red', alpha=0.7)
    ax1.set_title('Quantum Bit Error Rate (QBER) vs Attack Scenarios')
    ax1.set_ylabel('QBER')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0.11, color='black', linestyle='--', label='Security Threshold')
    ax1.legend()
    
    # Key Generation Rate vs Attack Scenarios
    ax2.bar(scenarios, key_rates, color='blue', alpha=0.7)
    ax2.set_title('Key Generation Rate vs Attack Scenarios')
    ax2.set_ylabel('Key Generation Rate')
    ax2.tick_params(axis='x', rotation=45)
    
    # Attack Success Rate
    ax3.bar(scenarios, attack_success_rates, color='orange', alpha=0.7)
    ax3.set_title('Attack Success Rate vs Attack Scenarios')
    ax3.set_ylabel('Attack Success Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # Information Gained by Eve
    ax4.bar(scenarios, information_gained, color='green', alpha=0.7)
    ax4.set_title('Information Gained by Eve vs Attack Scenarios')
    ax4.set_ylabel('Information Gained')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('quantum_encryption_verification/eavesdropping_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Eavesdropper Analysis for Quantum Key Distribution")
    print("=" * 55)
    
    # Run eavesdropping analysis
    results = analyze_eavesdropping_scenarios()
    
    # Create visualizations
    plot_eavesdropping_analysis(results)
    
    # Print summary
    print("\nEavesdropping Analysis Summary:")
    print("-" * 40)
    for scenario, data in results.items():
        result = data['qkd_result']
        stats = data['attack_stats']
        print(f"{scenario:25} | QBER: {result.error_rate:.4f} | "
              f"Key Rate: {result.key_generation_rate:.4f} | "
              f"Secure: {not result.eavesdropping_detected}")
        if stats:
            print(f"{'':25} | Attack Success: {stats['attack_success_rate']:.4f} | "
                  f"Info Gained: {stats['information_gained']:.4f}")


