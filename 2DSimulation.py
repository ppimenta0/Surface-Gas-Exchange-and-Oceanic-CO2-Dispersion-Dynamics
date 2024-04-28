import numpy as np
from pyevtk.hl import gridToVTK
from fipy import CellVariable, TransientTerm, DiffusionTerm, AdvectionTerm, FaceVariable, Grid2D
import os
from fipy.tools.numerix import MA

# Create output directory for VTK files
vtk_dir = "Advection_vtk"
if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)

# Export function adapted for 2D data
def export_to_vtk(output_dir, step, variable1, mesh):
    # Generate coordinates for a rectilinear grid
    x = np.linspace(0, mesh.nx * mesh.dx, mesh.nx, dtype=np.float64)
    y = np.linspace(0, mesh.ny * mesh.dy, mesh.ny, dtype=np.float64)
    z = np.array([0.0], dtype=np.float64)  # Singleton Z dimension for 2D data
    
    # The variable (concentration) data should be reshaped to match VTK's expectation
    # For a 2D dataset in 3D space, we add an extra dimension to concentration
    concentration = variable1.reshape(mesh.nx, mesh.ny, 1)
    # Export to VTK
    filename = os.path.join(output_dir, f"concentration_step_{step}.vtr")
    gridToVTK(filename, x, y, z, pointData={"concentration": concentration,})


# Constants and initial conditions
C_air_initial = 280
C_sea_initial = 28.0
k = 0.06944
injection_time = 15.0
gas_injection = 280
new_C_air = C_air_initial + gas_injection

# 2D mesh setup
nx, ny = 5, 100  # Grid points in x and y directions
dx, dy = 0.3, 0.3  # Grid spacing in x and y directions
mesh = Grid2D(nx=nx, dx=dx, ny=ny, dy=dy)


# Cell variable for the solution
phi = CellVariable(name="solution variable", mesh=mesh, value=0.)

# Time settings
total_time = 1000
dt = 1
num_steps = int(total_time / dt)

# PDE definition
D0 = 1.9e-9
depth=ny*dy

def v(depth):
    if depth >3 and depth<5:
        return 0.15
    if depth > 12 and depth < 16:
        return 0.2
    else:
        return 0



def Dt(depth): 
    if depth <= 1.5:
        return 1
    else:
        return 0

depths = np.linspace(0, ny * dy, ny)

#Turb_diff = np.array([Dt(depth) for depth in mesh.cellCenters[0]])
#print(mesh.cellCenters[0])
advectionCoeffs_depth =np.array([v(depth) for depth in mesh.cellCenters[1]])
#advectionCoeffs = np.tile(advectionCoeffs_depth, (nx, 1)).T

advectionCoeffs_face = FaceVariable(mesh=mesh, value=advectionCoeffs_depth)


K = 1e-4
eq = TransientTerm() + AdvectionTerm(coeff=advectionCoeffs_face[:,0]) == DiffusionTerm(coeff=D0)

# Gas exchange flux functioncl
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

    phi.constrain(flux, where=mesh.exteriorFaces)
    eq.solve(var=phi, dt=dt)

    if step == 0:
        concentration_profiles[step, :, :] = phi.value.reshape((nx, ny))
    else:
        concentration_profiles[step, :, :] = concentration_profiles[step - 1, :] + phi.value.reshape((nx, ny))


    export_to_vtk(vtk_dir, step, concentration_profiles[step, :, :], mesh)
