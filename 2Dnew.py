import numpy as np
from pyevtk.hl import gridToVTK
from fipy import CellVariable, TransientTerm, DiffusionTerm, AdvectionTerm, FaceVariable, Grid2D
import os

# Create output directory for VTK files
vtk_dir = "vtk_output"
if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)

# Export function adapted for 2D data
def export_to_vtk(output_dir, step, variable, mesh):
    # Generate coordinates for a rectilinear grid
    x = np.linspace(0, mesh.nx * mesh.dx, mesh.nx, dtype=np.float64)
    y = np.linspace(0, mesh.ny * mesh.dy, mesh.ny, dtype=np.float64)
    z = np.array([0.0], dtype=np.float64)  # Singleton Z dimension for 2D data
    
    # The variable (concentration) data should be reshaped to match VTK's expectation
    # For a 2D dataset in 3D space, we add an extra dimension to concentration
    concentration = variable.reshape(mesh.nx, mesh.ny, 1)
    
    # Export to VTK
    filename = os.path.join(output_dir, f"concentration_step_{step}.vtr")
    gridToVTK(filename, x, y, z, pointData={"concentration": concentration})


# Constants and initial conditions
C_air_initial = 280
C_sea_initial = 28.0
k = 0.06944
injection_time = 15.0
gas_injection = 280
new_C_air = C_air_initial + gas_injection

# 2D mesh setup
nx, ny = 200, 100  # Grid points in x and y directions
dx, dy = 0.1, 0.1  # Grid spacing in x and y directions
mesh = Grid2D(nx=nx, dx=dx, ny=ny, dy=dy)

# Cell variable for the solution
phi = CellVariable(name="solution variable", mesh=mesh, value=0.)

# Time settings
total_time = 100
dt = .1
num_steps = int(total_time / dt)

# PDE definition
D0 = 1e-2
depth=ny*dy

nCells = mesh.numberOfCells


AdvectionCoeffs = [0.001, 0.1]  # Uniform small velocity in x and y

K = 1e-4
Dt = 1e-1
eq = TransientTerm() + AdvectionTerm(coeff=AdvectionCoeffs) == DiffusionTerm(coeff=D0 + Dt)

# Gas exchange flux function
def gas_exchange_flux(sea_concentration, air_concentration, k):
    return k * (air_concentration - sea_concentration)

# Simulation run
air_concentration = C_air_initial
sea_concentration = C_sea_initial
air2_concentration = new_C_air
concentration_profiles = np.zeros((num_steps, nx, ny))

for step in range(num_steps):
    current_time = step * dt
    flux_0 = gas_exchange_flux(sea_concentration, air_concentration, k)
    if current_time < injection_time:
        air_concentration -= flux_0*dt
        sea_concentration += flux_0*dt
    else:
        air_concentration = air2_concentration
        sea_concentration += flux_0*dt
    
    flux = gas_exchange_flux(sea_concentration, air_concentration, k)*dt
    phi.constrain(flux, where=mesh.facesTop)
    eq.solve(var=phi, dt=dt)

    if step == 0:
        concentration_profiles[step, :, :] = phi.value.reshape((nx, ny))
    else:
        concentration_profiles[step, :, :] = concentration_profiles[step - 1, :] + phi.value.reshape((nx, ny))


    export_to_vtk(vtk_dir, step, concentration_profiles[step, :, :], mesh)
