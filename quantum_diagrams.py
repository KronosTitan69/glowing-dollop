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


