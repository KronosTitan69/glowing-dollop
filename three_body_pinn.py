import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from manim import *
import warnings
warnings.filterwarnings('ignore')

class ThreeBodyPINN:
    def __init__(self, hidden_layers=4, neurons_per_layer=50, learning_rate=0.001):
        """
        Physics-Informed Neural Network for the Three-Body Problem
        
        Args:
            hidden_layers: Number of hidden layers in the neural network
            neurons_per_layer: Number of neurons per hidden layer
            learning_rate: Learning rate for optimization
        """
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        
        # Physical constants
        self.G = 1.0  # Gravitational constant (normalized)
        self.m1, self.m2, self.m3 = 1.0, 1.0, 1.0  # Masses (can be modified)
        
        # Build the neural network
        self.model = self._build_model()
        
        # Initialize arrays to store training history
        self.loss_history = []
        self.physics_loss_history = []
        self.data_loss_history = []
        
    def _build_model(self):
        """Build the neural network architecture"""
        inputs = keras.Input(shape=(1,))  # Time input
        
        # Hidden layers with tanh activation
        x = layers.Dense(self.neurons_per_layer, activation='tanh')(inputs)
        for _ in range(self.hidden_layers - 1):
            x = layers.Dense(self.neurons_per_layer, activation='tanh')(x)
        
        # Output layer: 18 outputs (x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3)
        outputs = layers.Dense(18, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def physics_loss(self, t_batch):
        """Calculate the physics-informed loss based on Newton's laws"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t_batch)
            
            # Get neural network predictions
            predictions = self.model(t_batch)
            
            # Extract positions and velocities
            x1, y1, z1 = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
            x2, y2, z2 = predictions[:, 3:4], predictions[:, 4:5], predictions[:, 5:6]
            x3, y3, z3 = predictions[:, 6:7], predictions[:, 7:8], predictions[:, 8:9]
            vx1, vy1, vz1 = predictions[:, 9:10], predictions[:, 10:11], predictions[:, 11:12]
            vx2, vy2, vz2 = predictions[:, 12:13], predictions[:, 13:14], predictions[:, 14:15]
            vx3, vy3, vz3 = predictions[:, 15:16], predictions[:, 16:17], predictions[:, 17:18]
            
            # Calculate first derivatives (velocities)
            dx1_dt = tape.gradient(x1, t_batch)
            dy1_dt = tape.gradient(y1, t_batch)
            dz1_dt = tape.gradient(z1, t_batch)
            dx2_dt = tape.gradient(x2, t_batch)
            dy2_dt = tape.gradient(y2, t_batch)
            dz2_dt = tape.gradient(z2, t_batch)
            dx3_dt = tape.gradient(x3, t_batch)
            dy3_dt = tape.gradient(y3, t_batch)
            dz3_dt = tape.gradient(z3, t_batch)
            
            # Calculate second derivatives (accelerations)
            dvx1_dt = tape.gradient(vx1, t_batch)
            dvy1_dt = tape.gradient(vy1, t_batch)
            dvz1_dt = tape.gradient(vz1, t_batch)
            dvx2_dt = tape.gradient(vx2, t_batch)
            dvy2_dt = tape.gradient(vy2, t_batch)
            dvz2_dt = tape.gradient(vz2, t_batch)
            dvx3_dt = tape.gradient(vx3, t_batch)
            dvy3_dt = tape.gradient(vy3, t_batch)
            dvz3_dt = tape.gradient(vz3, t_batch)
        
        # Calculate distances
        r12 = tf.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 + 1e-8)
        r13 = tf.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2 + 1e-8)
        r23 = tf.sqrt((x2-x3)**2 + (y2-y3)**2 + (z2-z3)**2 + 1e-8)
        
        # Calculate gravitational forces (accelerations)
        # For body 1
        ax1 = -self.G * self.m2 * (x1-x2) / r12**3 - self.G * self.m3 * (x1-x3) / r13**3
        ay1 = -self.G * self.m2 * (y1-y2) / r12**3 - self.G * self.m3 * (y1-y3) / r13**3
        az1 = -self.G * self.m2 * (z1-z2) / r12**3 - self.G * self.m3 * (z1-z3) / r13**3
        
        # For body 2
        ax2 = -self.G * self.m1 * (x2-x1) / r12**3 - self.G * self.m3 * (x2-x3) / r23**3
        ay2 = -self.G * self.m1 * (y2-y1) / r12**3 - self.G * self.m3 * (y2-y3) / r23**3
        az2 = -self.G * self.m1 * (z2-z1) / r12**3 - self.G * self.m3 * (z2-z3) / r23**3
        
        # For body 3
        ax3 = -self.G * self.m1 * (x3-x1) / r13**3 - self.G * self.m2 * (x3-x2) / r23**3
        ay3 = -self.G * self.m1 * (y3-y1) / r13**3 - self.G * self.m2 * (y3-y2) / r23**3
        az3 = -self.G * self.m1 * (z3-z1) / r13**3 - self.G * self.m2 * (z3-z2) / r23**3
        
        # Physics equations loss
        f1 = dx1_dt - vx1
        f2 = dy1_dt - vy1
        f3 = dz1_dt - vz1
        f4 = dx2_dt - vx2
        f5 = dy2_dt - vy2
        f6 = dz2_dt - vz2
        f7 = dx3_dt - vx3
        f8 = dy3_dt - vy3
        f9 = dz3_dt - vz3
        
        f10 = dvx1_dt - ax1
        f11 = dvy1_dt - ay1
        f12 = dvz1_dt - az1
        f13 = dvx2_dt - ax2
        f14 = dvy2_dt - ay2
        f15 = dvz2_dt - az2
        f16 = dvx3_dt - ax3
        f17 = dvy3_dt - ay3
        f18 = dvz3_dt - az3
        
        physics_loss = tf.reduce_mean(tf.square(f1) + tf.square(f2) + tf.square(f3) +
                                    tf.square(f4) + tf.square(f5) + tf.square(f6) +
                                    tf.square(f7) + tf.square(f8) + tf.square(f9) +
                                    tf.square(f10) + tf.square(f11) + tf.square(f12) +
                                    tf.square(f13) + tf.square(f14) + tf.square(f15) +
                                    tf.square(f16) + tf.square(f17) + tf.square(f18))
        
        return physics_loss
    
    def initial_condition_loss(self, t_init, initial_states):
        """Calculate loss for initial conditions"""
        pred_init = self.model(t_init)
        return tf.reduce_mean(tf.square(pred_init - initial_states))
    
    def total_loss(self, t_batch, t_init, initial_states, alpha=1.0, beta=100.0):
        """Calculate total loss combining physics and initial condition losses"""
        phys_loss = self.physics_loss(t_batch)
        init_loss = self.initial_condition_loss(t_init, initial_states)
        
        total = alpha * phys_loss + beta * init_loss
        return total, phys_loss, init_loss
    
    def train(self, t_data, initial_states, epochs=5000, batch_size=100):
        """Train the PINN"""
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Initial conditions
        t_init = tf.constant([[0.0]], dtype=tf.float32)
        initial_states = tf.constant([initial_states], dtype=tf.float32)
        
        print("Starting PINN training for Three-Body Problem...")
        
        for epoch in range(epochs):
            # Random sampling of time points
            idx = np.random.choice(len(t_data), min(batch_size, len(t_data)), replace=False)
            t_batch = tf.constant(t_data[idx].reshape(-1, 1), dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                total_loss_val, phys_loss_val, init_loss_val = self.total_loss(
                    t_batch, t_init, initial_states)
            
            # Compute gradients and update
            grads = tape.gradient(total_loss_val, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            # Store losses
            self.loss_history.append(total_loss_val.numpy())
            self.physics_loss_history.append(phys_loss_val.numpy())
            self.data_loss_history.append(init_loss_val.numpy())
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Total Loss = {total_loss_val:.6f}, \
                      f"Physics Loss = {phys_loss_val:.6f}, \
                      f"Initial Loss = {init_loss_val:.6f}")
    
    def predict(self, t_data):
        """Make predictions using the trained model"""
        t_tensor = tf.constant(t_data.reshape(-1, 1), dtype=tf.float32)
        predictions = self.model(t_tensor).numpy()
        return predictions
    
    def calculate_energy(self, positions, velocities):
        """Calculate total energy of the system"""
        x1, y1, z1 = positions[:, 0], positions[:, 1], positions[:, 2]
        x2, y2, z2 = positions[:, 3], positions[:, 4], positions[:, 5]
        x3, y3, z3 = positions[:, 6], positions[:, 7], positions[:, 8]
        
        vx1, vy1, vz1 = velocities[:, 0], velocities[:, 1], velocities[:, 2]
        vx2, vy2, vz2 = velocities[:, 3], velocities[:, 4], velocities[:, 5]
        vx3, vy3, vz3 = velocities[:, 6], velocities[:, 7], velocities[:, 8]
        
        # Kinetic energy
        KE = 0.5 * self.m1 * (vx1**2 + vy1**2 + vz1**2) + 
             0.5 * self.m2 * (vx2**2 + vy2**2 + vz2**2) + 
             0.5 * self.m3 * (vx3**2 + vy3**2 + vz3**2)
        
        # Potential energy
        r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        r13 = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
        r23 = np.sqrt((x2-x3)**2 + (y2-y3)**2 + (z2-z3)**2)
        
        PE = -self.G * (self.m1*self.m2/r12 + self.m1*self.m3/r13 + self.m2*self.m3/r23)
        
        return KE + PE
    
    def plot_accuracy_metrics(self):
        """Plot various accuracy metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss evolution
        axes[0, 0].plot(self.loss_history, label='Total Loss', linewidth=2)
        axes[0, 0].plot(self.physics_loss_history, label='Physics Loss', linewidth=2)
        axes[0, 0].plot(self.data_loss_history, label='Initial Condition Loss', linewidth=2)
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy conservation check
        t_test = np.linspace(0, 10, 1000)
        predictions = self.predict(t_test)
        positions = predictions[:, :9]
        velocities = predictions[:, 9:]
        
        energies = []
        for i in range(len(t_test)):
            energy = self.calculate_energy(positions[i:i+1], velocities[i:i+1])
            energies.append(energy[0])
        
        axes[0, 1].plot(t_test, energies, linewidth=2, color='red')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Total Energy')
        axes[0, 1].set_title('Energy Conservation Check')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy variation
        energy_variation = np.abs(np.array(energies) - energies[0]) / np.abs(energies[0]) * 100
        axes[1, 0].plot(t_test, energy_variation, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Energy Variation (%)')
        axes[1, 0].set_title('Relative Energy Variation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss convergence rate
        if len(self.loss_history) > 100:
            loss_gradient = np.gradient(np.log(self.loss_history[100:]))
            axes[1, 1].plot(loss_gradient, linewidth=2, color='purple')
            axes[1, 1].set_xlabel('Epoch (after 100)')
            axes[1, 1].set_ylabel('Log Loss Gradient')
            axes[1, 1].set_title('Loss Convergence Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Manim Animation Classes
class ThreeBodyAnimation(ThreeDScene):
    def __init__(self, pinn_model, **kwargs):
        super().__init__(**kwargs)
        self.pinn = pinn_model
        
    def construct(self):
        # Set up the 3D scene
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        
        # Generate trajectory data
        t_data = np.linspace(0, 10, 500)
        predictions = self.pinn.predict(t_data)
        
        # Extract positions for each body
        x1, y1, z1 = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        x2, y2, z2 = predictions[:, 3], predictions[:, 4], predictions[:, 5]
        x3, y3, z3 = predictions[:, 6], predictions[:, 7], predictions[:, 8]
        
        # Create spheres for the three bodies
        body1 = Sphere(radius=0.1, color=RED).move_to([x1[0], y1[0], z1[0]])
        body2 = Sphere(radius=0.1, color=BLUE).move_to([x2[0], y2[0], z2[0]])
        body3 = Sphere(radius=0.1, color=GREEN).move_to([x3[0], y3[0], z3[0]])
        
        # Create trajectory paths
        path1 = VMobject(color=RED, stroke_width=2)
        path2 = VMobject(color=BLUE, stroke_width=2)
        path3 = VMobject(color=GREEN, stroke_width=2)
        
        # Add objects to scene
        self.add(body1, body2, body3)
        self.add(path1, path2, path3)
        
        # Animation function
        def update_bodies(mob, alpha):
            idx = int(alpha * (len(t_data) - 1))
            
            # Update body positions
            body1.move_to([x1[idx], y1[idx], z1[idx]])
            body2.move_to([x2[idx], y2[idx], z2[idx]])
            body3.move_to([x3[idx], y3[idx], z3[idx]])
            
            # Update trajectory paths
            if idx > 0:
                points1 = [[x1[i], y1[i], z1[i]] for i in range(min(idx+1, len(x1)))]
                points2 = [[x2[i], y2[i], z2[i]] for i in range(min(idx+1, len(x2)))]
                points3 = [[x3[i], y3[i], z3[i]] for i in range(min(idx+1, len(x3)))]
                
                if len(points1) > 1:
                    path1.set_points_as_corners(points1)
                    path2.set_points_as_corners(points2)
                    path3.set_points_as_corners(points3)
        
        # Create and play animation
        animation_group = AnimationGroup(
            UpdateFromAlphaFunc(VGroup(body1, body2, body3, path1, path2, path3), update_bodies),
            run_time=10
        )
        
        self.play(animation_group)
        self.wait(2)

class EnergyVisualization(Scene):
    def __init__(self, pinn_model, **kwargs):
        super().__init__(**kwargs)
        self.pinn = pinn_model
        
    def construct(self):
        # Create title
        title = Text("Energy Conservation in Three-Body System", font_size=36)
        title.to_edge(UP)
        self.add(title)
        
        # Generate energy data
        t_data = np.linspace(0, 10, 200)
        predictions = self.pinn.predict(t_data)
        positions = predictions[:, :9]
        velocities = predictions[:, 9:]
        
        energies = []
        for i in range(len(t_data)):
            energy = self.pinn.calculate_energy(positions[i:i+1], velocities[i:i+1])
            energies.append(energy[0])
        
        # Create axes
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[min(energies)*1.1, max(energies)*1.1, (max(energies)-min(energies))/5],
            x_length=10,
            y_length=6,
            axis_config={"color": BLUE},
        )
        
        # Create energy curve
        energy_curve = axes.plot_line_graph(
            t_data, energies,
            line_color=RED,
            stroke_width=3
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("Time")
        y_label = axes.get_y_axis_label("Total Energy")
        
        # Add everything to scene
        self.add(axes, energy_curve, x_label, y_label)
        
        # Animate the curve drawing
        self.play(Create(axes))
        self.play(Create(energy_curve), run_time=3)
        self.wait(2)

def main():
    """Main function to demonstrate the Three-Body PINN"""
    print("Initializing Three-Body Physics-Informed Neural Network...")
    
    # Initialize the PINN
    pinn = ThreeBodyPINN(hidden_layers=4, neurons_per_layer=64, learning_rate=0.001)
    
    # Set up initial conditions (Figure-8 configuration)
    initial_conditions = [
        # Positions (x1, y1, z1, x2, y2, z2, x3, y3, z3)
        0.97000436, -0.24308753, 0.0,  # Body 1
        -0.97000436, 0.24308753, 0.0,  # Body 2
        0.0, 0.0, 0.0,                 # Body 3
        # Velocities (vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3)
        0.466203685, 0.43236573, 0.0,  # Body 1 velocity
        0.466203685, 0.43236573, 0.0,  # Body 2 velocity
        -0.93240737, -0.86473146, 0.0  # Body 3 velocity
    ]
    
    # Generate training time data
    t_train = np.linspace(0, 10, 2000)
    
    # Train the model
    pinn.train(t_train, initial_conditions, epochs=3000, batch_size=150)
    
    # Generate test data and predictions
    t_test = np.linspace(0, 15, 1500)
    predictions = pinn.predict(t_test)
    
    # Plot accuracy metrics
    print("Generating accuracy metrics...")
    pinn.plot_accuracy_metrics()
    
    # Create 3D trajectory plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    x1, y1, z1 = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    x2, y2, z2 = predictions[:, 3], predictions[:, 4], predictions[:, 5]
    x3, y3, z3 = predictions[:, 6], predictions[:, 7], predictions[:, 8]
    
    # Plot trajectories
    ax.plot(x1, y1, z1, 'r-', linewidth=2, label='Body 1', alpha=0.8)
    ax.plot(x2, y2, z2, 'b-', linewidth=2, label='Body 2', alpha=0.8)
    ax.plot(x3, y3, z3, 'g-', linewidth=2, label='Body 3', alpha=0.8)
    
    # Mark initial positions
    ax.scatter([x1[0]], [y1[0]], [z1[0]], color='red', s=100, marker='o')
    ax.scatter([x2[0]], [y2[0]], [z2[0]], color='blue', s=100, marker='o')
    ax.scatter([x3[0]], [y3[0]], [z3[0]], color='green', s=100, marker='o')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Three-Body Problem Solution using PINN')
    ax.legend()
    plt.show()
    
    # Interactive Plotly visualization
    fig_plotly = go.Figure()
    
    fig_plotly.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines',
        line=dict(color='red', width=4),
        name='Body 1'
    ))
    
    fig_plotly.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Body 2'
    ))
    
    fig_plotly.add_trace(go.Scatter3d(
        x=x3, y=y3, z=z3,
        mode='lines',
        line=dict(color='green', width=4),
        name='Body 3'
    ))
    
    fig_plotly.update_layout(
        title='Interactive Three-Body Problem Visualization',
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position'
        )
    )
    
    fig_plotly.show()
    
    # Generate Manim animations
    print("Creating Manim animations...")
    
    try:
        # Create animation scenes
        trajectory_scene = ThreeBodyAnimation(pinn)
        energy_scene = EnergyVisualization(pinn)
        
        print("Manim scenes created successfully!")
        print("To render animations, run:")
        print("manim three_body_pinn.py ThreeBodyAnimation -p")
        print("manim three_body_pinn.py EnergyVisualization -p")
        
    except Exception as e:
        print(f"Manim animation creation failed: {e}")
        print("Make sure Manim is properly installed: pip install manim")
    
    # Calculate and display final metrics
    print("\n" + "="*60)
    print("FINAL ACCURACY METRICS")
    print("="*60)
    
    # Energy conservation
    positions = predictions[:, :9]
    velocities = predictions[:, 9:]
    energies = [pinn.calculate_energy(positions[i:i+1], velocities[i:i+1])[0] 
               for i in range(len(t_test))]
    
    initial_energy = energies[0]
    final_energy = energies[-1]
    energy_drift = abs(final_energy - initial_energy) / abs(initial_energy) * 100
    
    print(f"Initial Energy: {initial_energy:.6f}")
    print(f"Final Energy: {final_energy:.6f}")
    print(f"Energy Drift: {energy_drift:.4f}%")
    print(f"Final Training Loss: {pinn.loss_history[-1]:.8f}")
    print(f"Final Physics Loss: {pinn.physics_loss_history[-1]:.8f}")
    
    if energy_drift < 1.0:
        print("✅ Excellent energy conservation!")
    elif energy_drift < 5.0:
        print("✅ Good energy conservation!")
    else:
        print("⚠️  Energy conservation could be improved")
    
    return pinn, predictions, t_test

if __name__ == "__main__":
    pinn_model, predictions, time_data = main()