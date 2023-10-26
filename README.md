# space-black-hole-project
This is space project repository
saniya's team
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Particle properties
particle_mass = 1.0  # Mass of the particle in kg
initial_position = np.array([10.0, 0, 0])  # Initial position (x, y, z) in meters
initial_velocity = np.array([0, 2000, 0])   # Initial velocity (vx, vy, vz) in m/s

# Simulation parameters
total_time = 10  # Total simulation time in seconds
num_steps = 1000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store particle positions
positions = [initial_position]

# Simulate particle motion
for _ in range(num_steps):
    # Calculate gravitational force (simplified)
    r = np.linalg.norm(positions[-1])  # Distance from black hole
    force = -G * M * particle_mass / r**2
    acceleration = force / particle_mass

    # Update velocity and position
    velocity = initial_velocity + acceleration * dt
    position = positions[-1] + velocity * dt

    # Append the new position to the list
    positions.append(position)

# Extract x, y, and z coordinates from positions
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]
z = [pos[2] for pos in positions]

# Plot the particle's path
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Particle Orbiting a Schwarzschild Black Hole')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Initial conditions
r0 = 10.0e3  # Initial radial position in meters
v0 = 1000.0  # Initial radial velocity in m/s

# Simulation parameters
total_time = 100  # Total simulation time in seconds
num_steps = 1000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store particle positions
r_values = [r0]

# Simulate particle's radial motion
for _ in range(num_steps):
    # Calculate the Schwarzschild radial geodesic equation
    r = r_values[-1]
    drdt = np.sqrt((2 * G * M) / r - (c**2) * (1 - (2 * G * M) / r))
    r_new = r + drdt * dt

    # Append the new radial position to the list
    r_values.append(r_new)

# Plot the particle's radial motion
t_values = [i * dt for i in range(num_steps + 1)]

plt.figure(figsize=(10, 6))
plt.plot(t_values, r_values)
plt.xlabel('Time (s)')
plt.ylabel('Radial Position (m)')
plt.title('Particle Radial Motion in Schwarzschild Coordinates')
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Initial conditions
r0 = 1000.0  # Initial radial position in meters
phi0 = np.pi / 4  # Initial angular position in radians
v_r0 = 0.0  # Initial radial velocity in m/s
v_phi0 = 1000.0  # Initial angular velocity in m/s

# Simulation parameters
total_time = 1000  # Total simulation time in seconds
num_steps = 10000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store particle positions
r_values = [r0]
phi_values = [phi0]

# Simulate particle's motion
for _ in range(num_steps):
    # Calculate the Schwarzschild radial geodesic equation
    r = r_values[-1]
    v_r = v_r0
    drdt = v_r

    # Calculate the angular geodesic equation
    phi = phi_values[-1]
    v_phi = v_phi0
    dphidt = v_phi / (r**2)

    # Update positions and velocities
    r_new = r + drdt * dt
    phi_new = phi + dphidt * dt

    # Append the new positions to the lists
    r_values.append(r_new)
    phi_values.append(phi_new)

# Convert polar coordinates to Cartesian for visualization
x_values = [r * np.cos(phi) for r, phi in zip(r_values, phi_values)]
y_values = [r * np.sin(phi) for r, phi in zip(r_values, phi_values)]

# Plot the particle's path
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Particle Motion in Schwarzschild Coordinates')
plt.grid(True)
plt.show()
import numpy as np

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Schwarzschild metric components
def schwarzschild_metric_components(r):
    g_tt = -(1 - 2 * G * M / (r * c**2))
    g_rr = 1 / (1 - 2 * G * M / (r * c**2))
    g_theta_theta = r**2
    g_phi_phi = (r * np.sin(np.pi / 2))**2  # Assuming θ = π/2 for simplicity
    
    return g_tt, g_rr, g_theta_theta, g_phi_phi

# Point at which to calculate metric components (choose r > 2GM/c^2)
r = 10000.0  # Radial position in meters

# Calculate metric components
g_tt, g_rr, g_theta_theta, g_phi_phi = schwarzschild_metric_components(r)

# Display the results
print(f"Schwarzschild Metric Components at r = {r} meters:")
print(f"g_tt: {g_tt}")
print(f"g_rr: {g_rr}")
print(f"g_θθ: {g_theta_theta}")
print(f"g_ϕϕ: {g_phi_phi}")
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Initial conditions
r0 = 10000.0  # Initial radial position in meters
v_r0 = 0.0    # Initial radial velocity in m/s

# Simulation parameters
total_time = 1000  # Total simulation time in seconds
num_steps = 10000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store particle positions
r_values = [r0]

# Simulate geodesics for radial motion
for _ in range(num_steps):
    # Calculate the Schwarzschild radial geodesic equation
    r = r_values[-1]
    v_r = v_r0
    drdt = v_r

    # Update radial position
    r_new = r + drdt * dt

    # Append the new radial position to the list
    r_values.append(r_new)

# Plot the particle's radial motion
t_values = [i * dt for i in range(num_steps + 1)]

plt.figure(figsize=(10, 6))
plt.plot(t_values, r_values)
plt.xlabel('Time (s)')
plt.ylabel('Radial Position (m)')
plt.title('Geodesics for Radial Motion in Schwarzschild Coordinates')
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Initial conditions
r0 = 10000.0  # Initial radial position in meters
v_r0 = 0.0    # Initial radial velocity in m/s

# Simulation parameters
total_time = 1000  # Total simulation time in seconds
num_steps = 10000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store particle positions
r_values = [r0]

# Numerical integration using the Euler method
for _ in range(num_steps):
    # Calculate the Schwarzschild radial geodesic equation
    r = r_values[-1]
    v_r = v_r0
    drdt = v_r

    # Update radial position using Euler method
    r_new = r + drdt * dt

    # Append the new radial position to the list
    r_values.append(r_new)

# Convert to numpy arrays for easier calculations
r_values = np.array(r_values)

# Plot the particle's radial motion
t_values = np.linspace(0, total_time, num_steps + 1)

plt.figure(figsize=(10, 6))
plt.plot(t_values, r_values)
plt.xlabel('Time (s)')
plt.ylabel('Radial Position (m)')
plt.title('Geodesics for Radial Motion in Schwarzschild Coordinates (Numerical Integration)')
plt.grid(True)
plt.show()
import numpy as np

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Initial conditions
r0 = 10000.0  # Initial radial position in meters
v_r0 = 1000.0  # Initial radial velocity in m/s

# Simulation parameters
total_time = 1000  # Total simulation time in seconds
num_steps = 10000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store particle positions and velocities
r_values = [r0]
v_r_values = [v_r0]

# Numerical integration using the Euler method
for _ in range(num_steps):
    # Calculate the Schwarzschild radial geodesic equation
    r = r_values[-1]
    v_r = v_r_values[-1]
    drdt = v_r

    # Update radial position using Euler method
    r_new = r + drdt * dt

    # Append the new radial position to the list
    r_values.append(r_new)

    # No change in velocity for this simplified radial motion
    v_r_values.append(v_r)

# Convert to numpy arrays for easier calculations
r_values = np.array(r_values)
v_r_values = np.array(v_r_values)

# Plot the particle's radial motion
t_values = np.linspace(0, total_time, num_steps + 1)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(t_values, r_values)
plt.xlabel('Time (s)')
plt.ylabel('Radial Position (m)')
plt.title('Geodesics for Radial Motion in Schwarzschild Coordinates (Numerical Integration)')
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Initial conditions
r0 = 10000.0  # Initial radial position in meters
phi0 = np.pi / 2  # Initial azimuthal angle in radians
v_r0 = c  # Initial radial velocity (speed of light)
v_phi0 = 0.0  # Initial angular velocity

# Simulation parameters
total_time = 0.1  # Total simulation time in seconds
num_steps = 1000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store photon positions
r_values = [r0]
phi_values = [phi0]

# Numerical integration of photon geodesics
for _ in range(num_steps):
    r = r_values[-1]
    phi = phi_values[-1]
    v_r = v_r0
    v_phi = v_phi0

    # Calculate photon geodesics in Schwarzschild spacetime
    drdt = v_r
    dphidt = v_phi / r**2

    # Update positions using numerical integration
    r_new = r + drdt * dt
    phi_new = phi + dphidt * dt

    # Append the new positions to the lists
    r_values.append(r_new)
    phi_values.append(phi_new)

# Convert polar coordinates to Cartesian for visualization
x_values = [r * np.cos(phi) for r, phi in zip(r_values, phi_values)]
y_values = [r * np.sin(phi) for r, phi in zip(r_values, phi_values)]

# Plot the photon paths
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Light Paths (Photon Geodesics) in Schwarzschild Coordinates')
plt.grid(True)
plt.show()
import vtkmodules.all as vtk
import numpy as np

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Create a VTK scene
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Create a sphere for the black hole
black_hole = vtk.vtkSphereSource()
black_hole.SetCenter(0, 0, 0)
black_hole.SetRadius(2 * G * M / (c**2))
black_hole_mapper = vtk.vtkPolyDataMapper()
black_hole_mapper.SetInputConnection(black_hole.GetOutputPort())
black_hole_actor = vtk.vtkActor()
black_hole_actor.SetMapper(black_hole_mapper)
renderer.AddActor(black_hole_actor)

# Create a particle trajectory (simplified)
particle_trajectory = vtk.vtkLineSource()
particle_trajectory.SetPoint1(10000, 0, 0)
particle_trajectory.SetPoint2(2 * G * M / (c**2), 0, 0)
trajectory_mapper = vtk.vtkPolyDataMapper()
trajectory_mapper.SetInputConnection(particle_trajectory.GetOutputPort())
trajectory_actor = vtk.vtkActor()
trajectory_actor.SetMapper(trajectory_mapper)
renderer.AddActor(trajectory_actor)

# Set up the caimport numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Initial conditions
r0 = 10000.0  # Initial radial position in meters
v_r0 = -c  # Initial radial velocity in m/s

# Simulation parameters
total_time = 100  # Total simulation time in seconds
num_steps = 1000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store particle positions and time values
r_values = [r0]
t_values = [0.0]

# Simulate particle motion including relativistic effects
for _ in range(num_steps):
    r = r_values[-1]
    v_r = v_r0

    # Calculate time dilation factor (Lorentz factor)
    gamma = 1 / np.sqrt(1 - (v_r**2 / c**2))

    # Calculate proper time experienced by the particle
    dtau = dt / gamma

    # Update time and radial position
    t_new = t_values[-1] + dtau
    r_new = r + v_r * dtau

    # Append the new radial position and time value to the lists
    r_values.append(r_new)
    t_values.append(t_new)

# Plot the particle's radial position including relativistic effects
plt.figure(figsize=(10, 6))
plt.plot(t_values, r_values)
plt.xlabel('Proper Time (s)')
plt.ylabel('Radial Position (m)')
plt.title('Particle Motion with Relativistic Effects in Schwarzschild Coordinates')
plt.grid(True)
plt.show()
mera and render
renderer.ResetCamera()
render_window.Render()
interactor.Start()
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light (m/s)
M = 1.989e30     # Mass of the black hole in kg (e.g., a solar mass)

# Initial conditions
r0 = 10000.0  # Initial radial position in meters
v_r0 = -c  # Initial radial velocity in m/s

# Simulation parameters
total_time = 100  # Total simulation time in seconds
num_steps = 1000  # Number of time steps

# Time step
dt = total_time / num_steps

# Lists to store particle positions and time values
r_values = [r0]
t_values = [0.0]

# Simulate particle motion including relativistic effects
for _ in range(num_steps):
    r = r_values[-1]
    v_r = v_r0

    # Calculate time dilation factor (Lorentz factor)
    gamma = 1 / np.sqrt(1 - (v_r**2 / c**2))

    # Calculate proper time experienced by the particle
    dtau = dt / gamma

    # Update time and radial position
    t_new = t_values[-1] + dtau
    r_new = r + v_r * dtau

    # Append the new radial position and time value to the lists
    r_values.append(r_new)
    t_values.append(t_new)

# Plot the particle's radial position including relativistic effects
plt.figure(figsize=(10, 6))
plt.plot(t_values, r_values)
plt.xlabel('Proper Time (s)')
plt.ylabel('Radial Position (m)')
plt.title('Particle Motion with Relativistic Effects in Schwarzschild Coordinates')
plt.grid(True)
plt.show()
import vtkmodules.all as vtk
from PyQt5.QtWidgets import QMainWindow, QVTKRenderWindowInteractor
from PyQt5.QtCore import QTimer

class BlackHoleSimulation(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a VTK scene
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        # Create a VTK interactor
        self.interactor = QVTKRenderWindowInteractor(self)
        self.interactor.SetRenderWindow(self.render_window)
        self.setCentralWidget(self.interactor)

        # Create a sphere for the black hole
        black_hole = vtk.vtkSphereSource()
        black_hole.SetCenter(0, 0, 0)
        black_hole.SetRadius(2 * G * M / (c**2))
        black_hole_mapper = vtk.vtkPolyDataMapper()
        black_hole_mapper.SetInputConnection(black_hole.GetOutputPort())
        black_hole_actor = vtk.vtkActor()
        black_hole_actor.SetMapper(black_hole_mapper)
        self.renderer.AddActor(black_hole_actor)

        # Create a particle trajectory (simplified)
        self.particle_trajectory = vtk.vtkLineSource()
        self.particle_trajectory.SetPoint1(10000, 0, 0)
        self.particle_trajectory.SetPoint2(2 * G * M / (c**2), 0, 0)
        trajectory_mapper = vtk.vtkPolyDataMapper()
        trajectory_mapper.SetInputConnection(self.particle_trajectory.GetOutputPort())
        self.trajectory_actor = vtk.vtkActor()
        self.trajectory_actor.SetMapper(trajectory_mapper)
        self.renderer.AddActor(self.trajectory_actor)

        # Set up the camera
        self.renderer.ResetCamera()

        # Create a timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.frame_rate = 30  # Frames per second
        self.timer.start(1000 / self.frame_rate)

    def update_simulation(self):
        # Update the particle trajectory (simplified)
        new_point2 = self.particle_trajectory.GetPoint2()
        new_point2 = (new_point2[0] - 100, new_point2[1], new_point2[2])
        self.particle_trajectory.SetPoint2(new_point2)
        self.trajectory_actor.Modified()

        # Update the render
        self.render_window.Render()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    G = 6.67430e-11
    c = 299792458.0
    M = 1.989e30

    app = QApplication(sys.argv)
    window = BlackHoleSimulation()
    window.show()
    sys.exit(app.exec_())
import vtkmodules.all as vtk
from PyQt5.QtWidgets import QMainWindow, QVTKRenderWindowInteractor
from PyQt5.QtCore import QTimer
import numpy as np
import matplotlib.pyplot as plt

class BlackHoleSimulation(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a VTK scene
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        # Create a VTK interactor
        self.interactor = QVTKRenderWindowInteractor(self)
        self.interactor.SetRenderWindow(self.render_window)
        self.setCentralWidget(self.interactor)

        # Create a sphere for the black hole
        black_hole = vtk.vtkSphereSource()
        black_hole.SetCenter(0, 0, 0)
        black_hole.SetRadius(2 * G * M / (c**2))
        black_hole_mapper = vtk.vtkPolyDataMapper()
        black_hole_mapper.SetInputConnection(black_hole.GetOutputPort())
        black_hole_actor = vtk.vtkActor()
        black_hole_actor.SetMapper(black_hole_mapper)
        self.renderer.AddActor(black_hole_actor)

        # Create a particle trajectory (simplified)
        self.particle_trajectory = vtk.vtkLineSource()
        self.particle_trajectory.SetPoint1(10000, 0, 0)
        self.particle_trajectory.SetPoint2(2 * G * M / (c**2), 0, 0)
        trajectory_mapper = vtk.vtkPolyDataMapper()
        trajectory_mapper.SetInputConnection(self.particle_trajectory.GetOutputPort())
        self.trajectory_actor = vtk.vtkActor()
        self.trajectory_actor.SetMapper(trajectory_mapper)
        self.renderer.AddActor(self.trajectory_actor)

        # Set up the camera
        self.renderer.ResetCamera()

        # Create a timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.frame_rate = 30  # Frames per second
        self.timer.start(1000 / self.frame_rate)

        # Initialize data storage
        self.time_values = []
        self.position_values = []

    def update_simulation(self):
        # Update the particle trajectory (simplified)
        new_point2 = self.particle_trajectory.GetPoint2()
        new_point2 = (new_point2[0] - 100, new_point2[1], new_point2[2])
        self.particle_trajectory.SetPoint2(new_point2)
        self.trajectory_actor.Modified()

        # Log data
        current_time = len(self.time_values) * (1.0 / self.frame_rate)
        self.time_values.append(current_time)
        self.position_values.append(new_point2[0])

        # Update the render
        self.render_window.Render()

    def save_data(self, filename):
        # Save the time and position data to a CSV file
        data = np.column_stack((self.time_values, self.position_values))
        np.savetxt(filename, data, delimiter=',', header='Time (s), Position (m)', comments='')

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    G = 6.67430e-11
    c = 299792458.0
    M = 1.989e30

    app = QApplication(sys.argv)
    window = BlackHoleSimulation()
    window.show()

    # Run the simulation for a specific duration (e.g., 10 seconds)
    QTimer.singleShot(10000, lambda: window.save_data('black_hole_simulation_data.csv'))

    sys.exit(app.exec_())
# Black Hole Simulation User Guide

Welcome to the Black Hole Simulation User Guide! This guide will help you get started with our black hole simulation project. This project is designed to provide a basic understanding of black hole physics 




