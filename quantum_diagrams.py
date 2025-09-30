"""
Quantum Concept Diagrams and Visualizations

This module creates comprehensive diagrams explaining quantum concepts including:
- Superposition states
- Entanglement
- No-cloning theorem
- Eavesdropping detection
- BB84 protocol flow
- Quantum measurement bases

Author: Quantum Encryption Verification System
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Qiskit integration for circuit visualizations
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.visualization import circuit_drawer, plot_histogram
    from qiskit.providers.basic_provider import BasicProvider
    from qiskit import execute
    QISKIT_AVAILABLE = True
    print("✓ Qiskit imported for quantum circuit visualizations")
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Circuit visualizations will be disabled.")

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class QuantumDiagramGenerator:
    """
    Generator for quantum concept diagrams and visualizations.
    
    This class creates educational diagrams explaining:
    - Quantum superposition
    - Entanglement
    - No-cloning theorem
    - BB84 protocol
    - Eavesdropping detection
    """
    
    def __init__(self):
        self.colors = {
            'alice': '#FF6B6B',
            'bob': '#4ECDC4', 
            'eve': '#FFE66D',
            'qubit': '#A8E6CF',
            'measurement': '#FF8B94',
            'success': '#95E1D3',
            'error': '#F38BA8'
        }
    
    def create_superposition_diagram(self) -> plt.Figure:
        """
        Create diagram explaining quantum superposition.
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Left plot: Classical bit states
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect('equal')
        
        # Classical states
        ax1.scatter([-1, 1], [0, 0], s=200, c=['red', 'blue'], alpha=0.8)
        ax1.text(-1, -0.3, '|0⟩', fontsize=16, ha='center', weight='bold')
        ax1.text(1, -0.3, '|1⟩', fontsize=16, ha='center', weight='bold')
        ax1.set_title('Classical Bit States', fontsize=14, weight='bold')
        ax1.set_xlabel('State Space')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Quantum superposition
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        
        # Superposition states
        theta = np.linspace(0, 2*np.pi, 100)
        circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
        
        # Show superposition as vector
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        for angle in angles:
            x = np.cos(angle)
            y = np.sin(angle)
            ax2.arrow(0, 0, x*0.8, y*0.8, head_width=0.1, head_length=0.1, 
                     fc='green', ec='green', alpha=0.7)
        
        ax2.scatter([-1, 1], [0, 0], s=200, c=['red', 'blue'], alpha=0.8)
        ax2.text(-1, -0.3, '|0⟩', fontsize=16, ha='center', weight='bold')
        ax2.text(1, -0.3, '|1⟩', fontsize=16, ha='center', weight='bold')
        ax2.text(0, 1.3, '|+⟩ = (|0⟩ + |1⟩)/√2', fontsize=12, ha='center', weight='bold')
        ax2.text(0, -1.3, '|-⟩ = (|0⟩ - |1⟩)/√2', fontsize=12, ha='center', weight='bold')
        ax2.set_title('Quantum Superposition States', fontsize=14, weight='bold')
        ax2.set_xlabel('State Space')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_entanglement_diagram(self) -> plt.Figure:
        """
        Create diagram explaining quantum entanglement.
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create entangled qubits
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        
        # Alice's qubit
        alice_circle = Circle((0, 2), 0.3, color=self.colors['alice'], alpha=0.8)
        ax.add_patch(alice_circle)
        ax.text(0, 2, 'A', fontsize=14, ha='center', va='center', weight='bold')
        
        # Bob's qubit
        bob_circle = Circle((0, 0), 0.3, color=self.colors['bob'], alpha=0.8)
        ax.add_patch(bob_circle)
        ax.text(0, 0, 'B', fontsize=14, ha='center', va='center', weight='bold')
        
        # Entanglement connection
        ax.plot([0, 0], [0.3, 1.7], 'k--', linewidth=3, alpha=0.7)
        ax.text(0.5, 1, 'Entangled\n|ψ⟩ = (|00⟩ + |11⟩)/√2', 
                fontsize=12, ha='left', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Measurement outcomes
        ax.text(-1.5, 2.5, 'Alice measures\n|0⟩ or |1⟩', fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['alice'], alpha=0.7))
        ax.text(-1.5, 0.5, 'Bob measures\n|0⟩ or |1⟩', fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['bob'], alpha=0.7))
        
        # Correlation arrows
        ax.arrow(-0.8, 2, -0.4, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.arrow(-0.8, 0, -0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        ax.set_title('Quantum Entanglement: Bell State', fontsize=16, weight='bold')
        ax.set_xlabel('Measurement Correlation')
        ax.grid(True, alpha=0.3)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_no_cloning_diagram(self) -> plt.Figure:
        """
        Create diagram explaining the no-cloning theorem.
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Left plot: Classical copying (allowed)
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-1, 3)
        ax1.set_aspect('equal')
        
        # Original classical bit
        ax1.scatter([0], [2], s=200, c='red', alpha=0.8)
        ax1.text(0, 1.7, 'Original\nClassical Bit', fontsize=10, ha='center')
        
        # Copying process
        ax1.arrow(0, 1.7, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax1.text(0.3, 1.2, 'Copy', fontsize=10, ha='left')
        
        # Copied classical bit
        ax1.scatter([0], [0], s=200, c='red', alpha=0.8)
        ax1.text(0, -0.3, 'Perfect Copy\nClassical Bit', fontsize=10, ha='center')
        
        ax1.set_title('Classical Copying (Allowed)', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Quantum no-cloning (forbidden)
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-1, 3)
        ax2.set_aspect('equal')
        
        # Original quantum state
        ax2.scatter([0], [2], s=200, c='blue', alpha=0.8)
        ax2.text(0, 1.7, 'Original\nQuantum State', fontsize=10, ha='center')
        
        # Forbidden copying process
        ax2.arrow(0, 1.7, 0, -0.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax2.text(0.3, 1.2, 'Copy?', fontsize=10, ha='left', color='red')
        
        # Failed copy
        ax2.scatter([0], [0], s=200, c='gray', alpha=0.5)
        ax2.text(0, -0.3, 'No Perfect Copy\nPossible', fontsize=10, ha='center', color='red')
        
        # Cross out the process
        ax2.plot([-0.3, 0.3], [1.2, 0.8], 'r-', linewidth=3)
        ax2.plot([0.3, -0.3], [1.2, 0.8], 'r-', linewidth=3)
        
        ax2.set_title('Quantum No-Cloning (Forbidden)', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_bb84_protocol_diagram(self) -> plt.Figure:
        """
        Create diagram showing BB84 protocol flow.
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Set up the diagram
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 6)
        ax.set_aspect('equal')
        
        # Alice
        alice_box = FancyBboxPatch((-0.5, 3), 1, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=self.colors['alice'], alpha=0.8)
        ax.add_patch(alice_box)
        ax.text(0, 3.75, 'Alice', fontsize=14, ha='center', weight='bold')
        
        # Bob
        bob_box = FancyBboxPatch((6.5, 3), 1, 1.5, boxstyle="round,pad=0.1",
                                facecolor=self.colors['bob'], alpha=0.8)
        ax.add_patch(bob_box)
        ax.text(7, 3.75, 'Bob', fontsize=14, ha='center', weight='bold')
        
        # Eve (optional)
        eve_box = FancyBboxPatch((3, 0.5), 1, 1, boxstyle="round,pad=0.1",
                                facecolor=self.colors['eve'], alpha=0.6)
        ax.add_patch(eve_box)
        ax.text(3.5, 1, 'Eve\n(Eavesdropper)', fontsize=10, ha='center', weight='bold')
        
        # Quantum channel
        ax.plot([0.5, 6.5], [3.75, 3.75], 'k-', linewidth=3, alpha=0.7)
        ax.text(3.5, 4.2, 'Quantum Channel', fontsize=12, ha='center', weight='bold')
        
        # Classical channel
        ax.plot([0.5, 6.5], [2.5, 2.5], 'b--', linewidth=2, alpha=0.7)
        ax.text(3.5, 2.2, 'Classical Channel', fontsize=12, ha='center', weight='bold')
        
        # Protocol steps
        steps = [
            (1, 5.5, "1. Alice prepares\nrandom qubits"),
            (2, 4.5, "2. Alice sends qubits\nthrough quantum channel"),
            (3, 3.5, "3. Bob measures qubits\nin random bases"),
            (4, 2.5, "4. Alice and Bob\ncompare bases"),
            (5, 1.5, "5. Sift matching bases\nand create raw key"),
            (6, 0.5, "6. Error correction\nand privacy amplification")
        ]
        
        for i, (step_num, y_pos, text) in enumerate(steps):
            # Step number
            ax.text(0.2, y_pos, f"{step_num}.", fontsize=12, ha='center', weight='bold')
            
            # Step description
            ax.text(1.5, y_pos, text, fontsize=10, ha='left', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
            
            # Arrow to next step
            if i < len(steps) - 1:
                ax.arrow(1.5, y_pos - 0.3, 0, -0.4, head_width=0.1, head_length=0.1,
                        fc='black', ec='black', alpha=0.7)
        
        # Quantum states representation
        qubit_positions = [(1.5, 4.2), (2.5, 4.2), (3.5, 4.2), (4.5, 4.2), (5.5, 4.2)]
        for i, (x, y) in enumerate(qubit_positions):
            circle = Circle((x, y), 0.1, color=self.colors['qubit'], alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y-0.3, f'|ψ{i+1}⟩', fontsize=8, ha='center')
        
        ax.set_title('BB84 Quantum Key Distribution Protocol', fontsize=16, weight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_measurement_bases_diagram(self) -> plt.Figure:
        """
        Create diagram showing quantum measurement bases.
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Left plot: Z-basis (computational basis)
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect('equal')
        
        # Z-basis states
        ax1.scatter([-1, 1], [0, 0], s=200, c=['red', 'blue'], alpha=0.8)
        ax1.text(-1, -0.3, '|0⟩', fontsize=16, ha='center', weight='bold')
        ax1.text(1, -0.3, '|1⟩', fontsize=16, ha='center', weight='bold')
        ax1.text(0, 1.2, 'Z-basis (Computational)', fontsize=14, ha='center', weight='bold')
        ax1.set_title('Z-basis Measurement', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: X-basis (Hadamard basis)
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        
        # X-basis states
        ax2.scatter([0, 0], [-1, 1], s=200, c=['green', 'orange'], alpha=0.8)
        ax2.text(-0.3, -1, '|-⟩', fontsize=16, ha='center', weight='bold')
        ax2.text(-0.3, 1, '|+⟩', fontsize=16, ha='center', weight='bold')
        ax2.text(0, -1.2, 'X-basis (Hadamard)', fontsize=14, ha='center', weight='bold')
        ax2.set_title('X-basis Measurement', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add superposition states
        theta = np.linspace(0, 2*np.pi, 100)
        circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
        ax2.add_patch(circle)
        
        plt.tight_layout()
        return fig
    
    def create_eavesdropping_detection_diagram(self) -> plt.Figure:
        """
        Create diagram showing eavesdropping detection in QKD.
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Set up the diagram
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 6)
        ax.set_aspect('equal')
        
        # Alice
        alice_box = FancyBboxPatch((-0.5, 4), 1, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=self.colors['alice'], alpha=0.8)
        ax.add_patch(alice_box)
        ax.text(0, 4.75, 'Alice', fontsize=14, ha='center', weight='bold')
        
        # Bob
        bob_box = FancyBboxPatch((6.5, 4), 1, 1.5, boxstyle="round,pad=0.1",
                                facecolor=self.colors['bob'], alpha=0.8)
        ax.add_patch(bob_box)
        ax.text(7, 4.75, 'Bob', fontsize=14, ha='center', weight='bold')
        
        # Eve
        eve_box = FancyBboxPatch((3, 2), 1, 1, boxstyle="round,pad=0.1",
                                facecolor=self.colors['eve'], alpha=0.8)
        ax.add_patch(eve_box)
        ax.text(3.5, 2.5, 'Eve', fontsize=12, ha='center', weight='bold')
        
        # Quantum channel (original)
        ax.plot([0.5, 3], [4.75, 2.5], 'k-', linewidth=3, alpha=0.7)
        ax.plot([3, 6.5], [2.5, 4.75], 'k-', linewidth=3, alpha=0.7)
        
        # Eavesdropping process
        ax.text(1.5, 3.5, 'Eve intercepts\nand measures', fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['eve'], alpha=0.7))
        
        # Error introduction
        ax.text(4.5, 3.5, 'Eve introduces\nerrors', fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['error'], alpha=0.7))
        
        # Detection process
        detection_steps = [
            (1, 1.5, "1. Alice and Bob\ncompare sample bits"),
            (2, 1, "2. Calculate QBER\n(Quantum Bit Error Rate)"),
            (3, 0.5, "3. If QBER > threshold:\nEavesdropping detected!"),
            (4, 0, "4. Abort protocol\nif compromised")
        ]
        
        for step_num, y_pos, text in detection_steps:
            ax.text(0.2, y_pos, f"{step_num}.", fontsize=12, ha='center', weight='bold')
            ax.text(1.5, y_pos, text, fontsize=10, ha='left', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        
        # QBER threshold
        ax.text(4, 1.5, 'QBER Threshold: 11%', fontsize=12, ha='center', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Security guarantee
        ax.text(4, 1, 'Security Guarantee:\nNo information leakage\nif QBER < threshold', 
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        ax.set_title('Eavesdropping Detection in QKD', fontsize=16, weight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_bb84_quantum_circuits_diagram(self) -> plt.Figure:
        """
        Create diagram showing BB84 quantum circuits using Qiskit.
        
        Returns:
            Figure with BB84 quantum circuit visualizations
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BB84 Quantum Circuits', fontsize=16, weight='bold')
        
        if not QISKIT_AVAILABLE:
            # Fallback diagram without Qiskit
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, 'Qiskit not available\nCircuit visualization disabled', 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            plt.tight_layout()
            return fig
        
        try:
            # Create BB84 state preparation circuits
            circuits = []
            titles = []
            
            # |0⟩ state (bit=0, basis=Z)
            qc_0z = QuantumCircuit(1, 1)
            qc_0z.measure(0, 0)
            circuits.append(qc_0z)
            titles.append('|0⟩ state (bit=0, Z basis)')
            
            # |1⟩ state (bit=1, basis=Z)
            qc_1z = QuantumCircuit(1, 1)
            qc_1z.x(0)
            qc_1z.measure(0, 0)
            circuits.append(qc_1z)
            titles.append('|1⟩ state (bit=1, Z basis)')
            
            # |+⟩ state (bit=0, basis=X)
            qc_0x = QuantumCircuit(1, 1)
            qc_0x.h(0)
            qc_0x.measure(0, 0)
            circuits.append(qc_0x)
            titles.append('|+⟩ state (bit=0, X basis)')
            
            # |-⟩ state (bit=1, basis=X)
            qc_1x = QuantumCircuit(1, 1)
            qc_1x.x(0)
            qc_1x.h(0)
            qc_1x.measure(0, 0)
            circuits.append(qc_1x)
            titles.append('|-⟩ state (bit=1, X basis)')
            
            # Draw circuits on subplots
            for i, (qc, title) in enumerate(zip(circuits, titles)):
                ax = axes.flat[i]
                
                # Create text representation of circuit
                circuit_str = self._quantum_circuit_to_text(qc)
                
                ax.text(0.1, 0.7, title, fontsize=12, weight='bold', transform=ax.transAxes)
                ax.text(0.1, 0.3, circuit_str, fontsize=10, family='monospace', 
                       transform=ax.transAxes, verticalalignment='top')
                
                # Add quantum state notation
                if 'Z basis' in title:
                    state_text = '|0⟩' if 'bit=0' in title else '|1⟩'
                else:  # X basis
                    state_text = '|+⟩' if 'bit=0' in title else '|-⟩'
                
                ax.text(0.8, 0.5, f'State: {state_text}', fontsize=14, weight='bold',
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['qubit']))
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
        
        except Exception as e:
            # Fallback if circuit creation fails
            print(f"Warning: Circuit visualization failed: {e}")
            for ax in axes.flat:
                ax.text(0.5, 0.5, f'Circuit visualization failed:\n{str(e)}', 
                       ha='center', va='center', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_quantum_measurement_diagram(self) -> plt.Figure:
        """
        Create diagram showing quantum measurement process.
        
        Returns:
            Figure explaining quantum measurement with circuits
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Quantum Measurement in Different Bases', fontsize=16, weight='bold')
        
        if QISKIT_AVAILABLE:
            try:
                # Demonstrate measurement in different bases
                test_cases = [
                    ('|0⟩ in Z basis', 'Measure |0⟩ in computational basis', 0, 'Z', 'Z'),
                    ('|0⟩ in X basis', 'Measure |0⟩ in Hadamard basis', 0, 'Z', 'X'),
                    ('|+⟩ in Z basis', 'Measure |+⟩ in computational basis', 0, 'X', 'Z'),
                    ('|+⟩ in X basis', 'Measure |+⟩ in Hadamard basis', 0, 'X', 'X'),
                ]
                
                for i, (title, desc, bit, prep_basis, meas_basis) in enumerate(test_cases):
                    ax = axes.flat[i]
                    
                    # Create circuit
                    qc = QuantumCircuit(1, 1)
                    
                    # Prepare state
                    if bit == 1:
                        qc.x(0)
                    if prep_basis == 'X':
                        qc.h(0)
                    
                    qc.barrier()
                    
                    # Measurement basis transformation
                    if meas_basis == 'X':
                        qc.h(0)
                    
                    qc.measure(0, 0)
                    
                    # Text representation
                    circuit_text = self._quantum_circuit_to_text(qc)
                    
                    ax.text(0.05, 0.9, title, fontsize=12, weight='bold', transform=ax.transAxes)
                    ax.text(0.05, 0.7, desc, fontsize=10, transform=ax.transAxes)
                    ax.text(0.05, 0.4, circuit_text, fontsize=9, family='monospace', 
                           transform=ax.transAxes, verticalalignment='top')
                    
                    # Expected outcome
                    if prep_basis == meas_basis:
                        outcome = "Deterministic result"
                        color = self.colors['success']
                    else:
                        outcome = "Random result (50/50)"
                        color = self.colors['error']
                    
                    ax.text(0.05, 0.1, f"Outcome: {outcome}", fontsize=10, weight='bold',
                           transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
            
            except Exception as e:
                print(f"Warning: Measurement diagram creation failed: {e}")
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'Quantum measurement\nvisualization unavailable', 
                           ha='center', va='center', fontsize=12)
                    ax.axis('off')
        else:
            # Fallback without Qiskit
            measurement_concepts = [
                "Computational Basis (Z)\n|0⟩ → 0, |1⟩ → 1",
                "Hadamard Basis (X)\n|+⟩ → 0, |-⟩ → 1",
                "Incompatible Bases\nMeasuring |0⟩ in X basis\ngives random result",
                "BB84 Security\nEve's measurement\ndisturbs quantum states"
            ]
            
            for i, concept in enumerate(measurement_concepts):
                ax = axes.flat[i]
                ax.text(0.5, 0.5, concept, ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _quantum_circuit_to_text(self, circuit) -> str:
        """
        Convert quantum circuit to simple text representation.
        
        Args:
            circuit: Qiskit QuantumCircuit
            
        Returns:
            Text representation of the circuit
        """
        if not QISKIT_AVAILABLE:
            return "Circuit visualization not available"
        
        try:
            # Simple text representation
            lines = []
            lines.append("q_0: ─")
            
            for instruction in circuit.data:
                gate_name = instruction.operation.name
                if gate_name == 'x':
                    lines[0] += "─X─"
                elif gate_name == 'h':
                    lines[0] += "─H─"
                elif gate_name == 'y':
                    lines[0] += "─Y─"
                elif gate_name == 'z':
                    lines[0] += "─Z─"
                elif gate_name == 'measure':
                    lines[0] += "─M─"
                elif gate_name == 'barrier':
                    lines[0] += "░░░"
                else:
                    lines[0] += f"─{gate_name.upper()}─"
            
            lines[0] += "─"
            
            if circuit.num_clbits > 0:
                lines.append("c_0: ═" + "═" * (len(lines[0]) - 5) + "═")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Circuit: {circuit.num_qubits} qubits, {len(circuit.data)} operations"
    
    def create_all_diagrams(self) -> List[plt.Figure]:
        """
        Create all quantum concept diagrams.
        
        Returns:
            List of matplotlib figures
        """
        figures = []
        
        print("Creating quantum superposition diagram...")
        figures.append(self.create_superposition_diagram())
        
        print("Creating entanglement diagram...")
        figures.append(self.create_entanglement_diagram())
        
        print("Creating no-cloning theorem diagram...")
        figures.append(self.create_no_cloning_diagram())
        
        print("Creating BB84 protocol diagram...")
        figures.append(self.create_bb84_protocol_diagram())
        
        print("Creating measurement bases diagram...")
        figures.append(self.create_measurement_bases_diagram())
        
        print("Creating eavesdropping detection diagram...")
        figures.append(self.create_eavesdropping_detection_diagram())
        
        # Add quantum circuit diagrams if Qiskit is available
        if QISKIT_AVAILABLE:
            print("Creating BB84 quantum circuits diagram...")
            figures.append(self.create_bb84_quantum_circuits_diagram())
            
            print("Creating quantum measurement diagram...")
            figures.append(self.create_quantum_measurement_diagram())
        else:
            print("Skipping quantum circuit diagrams (Qiskit not available)")
        
        return figures
    
    def save_all_diagrams(self, figures: List[plt.Figure], prefix: str = "quantum_diagram"):
        """
        Save all diagrams to files.
        
        Args:
            figures: List of matplotlib figures
            prefix: Prefix for filenames
        """
        diagram_names = [
            "superposition",
            "entanglement", 
            "no_cloning",
            "bb84_protocol",
            "measurement_bases",
            "eavesdropping_detection"
        ]
        
        # Add quantum circuit diagram names if Qiskit is available
        if QISKIT_AVAILABLE:
            diagram_names.extend([
                "bb84_quantum_circuits",
                "quantum_measurement"
            ])
        
        for i, (fig, name) in enumerate(zip(figures, diagram_names)):
            filename = f"quantum_encryption_verification/{prefix}_{name}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close(fig)

def create_comprehensive_visualizations():
    """
    Create comprehensive visualizations for quantum encryption verification.
    """
    print("Creating comprehensive quantum encryption visualizations...")
    
    # Create diagram generator
    diagram_gen = QuantumDiagramGenerator()
    
    # Generate all diagrams
    figures = diagram_gen.create_all_diagrams()
    
    # Save all diagrams
    diagram_gen.save_all_diagrams(figures)
    
    print("All quantum concept diagrams created and saved!")

if __name__ == "__main__":
    print("Quantum Concept Diagrams Generator")
    print("=" * 40)
    
    # Create all diagrams
    create_comprehensive_visualizations()
    
    print("\nQuantum concept diagrams generation completed!")
    print("Check the 'quantum_encryption_verification' directory for saved diagrams.")


