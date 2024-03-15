# import needed modules
import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm, AdvectionTerm

# Define constants and initial conditions
C_air_initial = 280  # Initial CO2 concentration in air
C_sea_initial = 28.0  # Initial CO2 concentration in sea
k = 0.06944  # Gas transfer velocity
injection_time = 15.0  # Time at which CO2 is injected
gas_injection = 280  # Amount of CO2 injected
new_C_air = C_air_initial + gas_injection  # New concentration of CO2 in air after injection

# Create a 1D mesh
nz = 150  # Number of grid points for depth
dz = 0.1  # Grid spacing for depth
Depth = nz*dz
mesh = Grid1D(nx=nz, dx=dz)
print(mesh.cellCenters[0][1])

# Create a cell variable to hold the solution
phi = CellVariable(name="solution variable", mesh=mesh, value=0.)

# Time grid
total_time = 500 # Total simulation time
dt = 1  # Time step size
num_steps = int(total_time / dt)  # Number of time steps

# Initialize variables for storing simulation data
times = np.arange(0, total_time, dt)
concentration_profiles = np.zeros((num_steps, nz))
T, Z = np.meshgrid(times, np.linspace(0, nz * dz, nz))

# Define the PDE
D0 = 1e-2  # Surface diffusion coefficient
K = 1e-4  # Depth-dependent diffusion coefficient
i=0


def v(depth):
    return 0.004 * depth ** 2


#print(v(mesh.cellCenters[0]))

#arrray = np.array([vvv(x) for x in mesh.cellCenters[0]])

#print(arrray)

#for depth in mesh.cellCenters[0]:print(depth)

advectionCoeffs = np.array([v(depth) for depth in mesh.cellCenters[0]])



D0=7.23e-9
#D = D0 + K * mesh.cellCenters[0]  # Diffusion coefficient varies with depth
Dt = 1e-1  # Turbulence diffusivity
eq = TransientTerm() + AdvectionTerm(coeff=advectionCoeffs)== DiffusionTerm(coeff=D0 + Dt)

# Function to calculate gas exchange flux
def gas_exchange_flux(sea_concentration, air_concentration, k):
    
    return k * (air_concentration - sea_concentration)



# Run the simulation
air_concentration = C_air_initial
sea_concentration = C_sea_initial
air2_concentration = new_C_air
flux_counter = 0

for step in range(num_steps):
    current_time = step * dt
    
    flux_0 = gas_exchange_flux(sea_concentration, air_concentration, k)
    # Update the air concentration after injection
    if current_time < injection_time:
        air_concentration -= flux_0*dt
        sea_concentration += flux_0*dt
    else:
        air_concentration = air2_concentration
        sea_concentration += flux_0*dt
    
    flux_counter += flux_0*dt
    #flux = gas_exchange_flux(sea_concentration, air_concentration, k)
    # Apply the flux as a boundary condition
    flux= gas_exchange_flux(sea_concentration, air_concentration, k)*dt
    phi.constrain(flux, where=mesh.facesLeft)

    # Solve the PDE
    eq.solve(var=phi, dt=dt)
    print(step)
    if step == 0:
        concentration_profiles[step, :] = phi.value
    else:
        concentration_profiles[step, :] = concentration_profiles[step - 1, :] + phi.value
    

# Transpose concentration_profiles to match dimensions with T and Z for plotting
concentration_profiles = concentration_profiles.T
print("total gas dissolved is",  flux_counter)
# Create the heatmap
plt.figure(figsize=(10, 5))
heatmap = plt.pcolor(T, Z, concentration_profiles, shading='auto')
plt.colorbar(heatmap)

# Flip y-axis so that max depth is at the origin
plt.gca().invert_yaxis()

# Label axes
plt.xlabel('Time (s)')
plt.ylabel('Depth (m)')
plt.title('CO2 Concentration Heatmap')

# Show the plot
plt.show()