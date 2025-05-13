# Sofia Wind Farm AEP Calculation
import time
import numpy as np
import pyproj
import openmdao.api as om



import multiprocessing
import os

n_cpu = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = '8'

from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver


from py_wake.site import UniformWeibullSite
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
import matplotlib.pyplot as plt
from py_wake.deflection_models import GCLHillDeflection



from topfarm.cost_models.py_wake_wrapper import AEPCostModelComponent

from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.py_wake_wrapper import AEPCostModelComponent


# --- Turbine Coordinates (lat, lon) ---
utm_turbines = np.array([
    [55.010475325822028, 1.965338917437833],
    [55.020811492061398, 1.982164664488693],
    [55.030111763636455, 1.997788572464856],
    [55.039409877333213, 2.014013399978381],
    [55.048705833408007, 2.029637307954516],
    [55.058343805455280, 2.045862135468013],
    [55.067635367164996, 2.062086962981539],
    [55.077612790301373, 2.077710870957731],
    [55.086899878841962, 2.093935698471228],
    [55.096528656042381, 2.109559606447419],
    [55.105811352366771, 2.126385353499643],
    [55.124370277907474, 2.158835008526665],
    [55.000825833175128, 1.982164664488693],
    [55.030111763636455, 2.063889721594933],
    [55.049738584240458, 2.095738457084622],
    [55.067979457657998, 2.127587192574310],
    [55.087243803672266, 2.158835008526665],
    [55.124713880404073, 2.190683744016326],
    [55.125057479944218, 2.223734318580625],
    [54.991863511277700, 1.998990411539523],
    [54.982899187411022, 2.015215239052992],
    [55.011509062352360, 2.096339376621984],
    [55.031144993937801, 2.128188112111587],
    [55.049738584240458, 2.160036847601333],
    [55.069011711384150, 2.192486502628327],
    [55.087243803672266, 2.224335238118016],
    [55.106842629991462, 2.256183973607648],
    [55.115779254598522, 2.239959146094122],
    [54.973243061020383, 2.031440066566518],
    [54.974622649826159, 2.096940296159232],
    [54.993931917089242, 2.128789031648921],
    [55.012542772235832, 2.161238686675944],
    [55.031489398118055, 2.192486502628327],
    [55.050427069999245, 2.224936157656742],
    [55.069355790042721, 2.257385812683708],
    [55.097560173211804, 2.273009720659900],
    [54.964619557155942, 2.048265813618741],
    [54.955304093290437, 2.063288802057599],
    [54.993931917089242, 2.193688341704416],
    [55.013231897354331, 2.226137996731353],
    [55.032178197598995, 2.257385812683708],
    [55.051459776442641, 2.289835467710731],
    [55.088275560415042, 2.288633628636035],
    [55.078988791343875, 2.306060295224256],
    [54.945986469496432, 2.080715468645764],
    [54.937357113973889, 2.096940296159232],
    [54.927690036835230, 2.113165123672758],
    [54.975657310328074, 2.161839606214727],
    [55.013576455472560, 2.290436387248036],
    [55.032866985239679, 2.322285122737725],
    [55.069699865742336, 2.322285122737725],
    [55.061097085647901, 2.337909030713917],
    [55.051804006005142, 2.353532938690080],
    [55.042508769081337, 2.370959605279666],
    [55.033555761040901, 2.386583513254436],
    [54.957029342141254, 2.194289261241693],
    [54.976002191237711, 2.226137996731353],
    [55.014610112063508, 2.354734777764719],
    [54.919402121254336, 2.129990870723617],
    [54.909730728454889, 2.146215698238535],
    [54.919747485157643, 2.194890180779026],
    [54.900748068305631, 2.161839606214727],
    [54.891417799836347, 2.179266272802863],
    [54.938737935320688, 2.226738916268687],
    [54.958070214134125, 2.259360899726630],
    [54.882436822880152, 2.195663428747281],
    [54.873102312234096, 2.211287336723444],
    [54.864457318970551, 2.227512164236941],
    [54.855464559347922, 2.244337911289193],
    [54.855464559347922, 2.276787566316216],
    [54.902135912719757, 2.292411474292379],
    [54.920789322658436, 2.259360899726630],
    [54.864111480680208, 2.294214232904380],
    [54.873793831612488, 2.310439060417849],
    [54.883128182108265, 2.326062968394041],
    [54.893151557947135, 2.341686876370261],
    [54.902135912719757, 2.357911703883730],
    [54.939434089119175, 2.291810554755045],
    [54.921134674655150, 2.324260209782068],
    [54.977042571711252, 2.290608715678985],
    [54.995661260007182, 2.323058370707372],
    [54.996005969087634, 2.387957680761417],
    [55.023917571161974, 2.402980669200275],
    [55.014960408419455, 2.419806416251078],
    [55.005311994920504, 2.436031243764575],
    [54.996350675207140, 2.452856990815434],
    [54.958415246077976, 2.323659290244706],
    [54.977387440723106, 2.355508025734451],
    [54.940124470059061, 2.356108945271728],
    [54.958415246077976, 2.389159519836085],
    [54.977387440723106, 2.420407335788440],
    [54.911463698250742, 2.374136531397198],
    [54.920789322658436, 2.389760439373362],
    [54.930458058166437, 2.405985266888337],
    [54.940124470059061, 2.422210094401834],
    [54.949443452603333, 2.437233082839242],
    [54.958415246077976, 2.454058829891522],
    [54.968074937681109, 2.470283657404991],
    [54.977732306772992, 2.485907565381183],
    [54.987387353638610, 2.469081818330352]
])

# --- Convert lat/lon to UTM (EPSG:32631) ---
transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:32631", always_xy=True)
x, y = transformer.transform(utm_turbines[:, 1], utm_turbines[:, 0])

# --- Wind Climate ---
f = np.array([7.17, 6.31, 6.93, 8.31, 8.27, 8.52, 9.2, 10.31, 10.35, 9.67, 8.57, 8.18])
a = np.array([9.64, 8.09, 9.24, 11.16, 11.15, 11.41, 12.39, 13.86, 13.99, 13.1, 11.41, 10.71])
k = np.array([2.146, 1.74, 2.338, 2.127, 2.053, 2.381, 1.826, 2.096, 2.361, 2.111, 2.088, 2.123])
site = UniformWeibullSite(p_wd=f/np.sum(f), a=a, k=k, ti=0.08)

# --- Turbine Model ---
wind_turbines = GenericWindTurbine(name="SG14-222DD", diameter=222, hub_height=141, power_norm=14000)


wf_model = Bastankhah_PorteAgel_2014(site, windTurbines=wind_turbines, k=0.0324555)



# site must include frequencies

#aep = wf_model.aep().sum()
aep = wf_model.aep(x, y).sum()


#print('AEP:', aep)
print('AEP:', aep)








# --- SOFIA WAKE STEERING OPTIMIZATION ---

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot

# --- Timing the optimization ---
start_time = time.time()

wd = np.arange(0, 360, 30)   # 18 directions
ws = np.arange(4, 26, 1)     # 11 speeds      




print("\nStarting Wake Steering Optimization (ILK)...")



# Reshape yaw: (n_wt, n_wd, n_ws)
n_wt = len(x)
n_ws = len(ws)
n_wd = len(wd)
#yaw_zero = np.zeros((n_wt, n_wd, n_ws))
yaw_init = np.random.uniform(-5, 5, size=(n_wt, n_wd, n_ws))

# Define AEP function with yaw array shape (n_wt, n_wd, n_ws)
# def aep_func(yaw_ilk):
#     simres = wf_model(x, y, wd=wd, ws=ws, yaw=yaw_ilk, tilt=0)
#     return simres.aep().sum()

aep_history = []


def aep_func(yaw_ilk):
    simres = wf_model(x, y, wd=wd, ws=ws, yaw=yaw_ilk, tilt=0)
    aep_val = simres.aep().sum()
    aep_history.append(aep_val)
    return aep_val



# Wrap cost component
cost_comp = CostModelComponent(
    input_keys=[('yaw_ilk', yaw_init)],
    n_wt=n_wt, 
    cost_function=aep_func,
    maximize=True,
    objective=True,
    output_keys=[('AEP', 0)]
)


from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver

driver = ScipyOptimizeDriver()

# Create the OpenMDAO driver directly

driver.options['optimizer'] = 'L-BFGS-B'
driver.options['maxiter'] = 500
driver.options['tol'] = 1e-6

# Use the OpenMDAO driver in the TopFarm problem
problem = TopFarmProblem(
    design_vars={'yaw_ilk': (yaw_init, -30, 30)},
    cost_comp=cost_comp,
    driver=driver,          # ← use the custom driver here
    plot_comp=NoPlot(),
    n_wt=n_wt
)




# Solve it
_, state, _ = problem.optimize()


# End timing
end_time = time.time()
elapsed_time = end_time - start_time


# Results
final_yaw_ilk = state['yaw_ilk']
final_aep = aep_func(final_yaw_ilk)

print("\n Wake Steering Optimization Complete")
print(f" Final AEP (GWh): {final_aep:.6f}")
print(f"⏱ Optimization Time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")









# --- Wake Map Visualization for multiple wind directions and speeds ---

# Define selected wind directions and wind speeds (by index)
selected_wd_indices = [0, 1, 2, 3]   # e.g., 0°, 30°, 60°, 90°
selected_ws_indices = [0, 3, 6, 10]  # e.g., 4, 10, 16, 24 m/s

for i, (wd_idx, ws_idx) in enumerate(zip(selected_wd_indices, selected_ws_indices), 1):
    simulation_result = wf_model(
        x, y,
        wd=wd[wd_idx],
        ws=ws[ws_idx],
        yaw=state['yaw_ilk'][:, wd_idx, ws_idx],  # use yaw for current direction/speed
        tilt=0
    )

    plt.figure(figsize=(12, 4))
    simulation_result.flow_map().plot_wake_map(cmap='jet')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f"Wake Map at wd={wd[wd_idx]}°, ws={ws[ws_idx]} m/s")
    plt.tight_layout()
    plt.show()










# Select turbine indices (e.g., first 10)
subset_indices = np.arange(10)  # or pick specific ones: [0, 5, 10, 15, ..., 90]

# Subset turbine positions
x_subset = x[subset_indices]
y_subset = y[subset_indices]

# Define selected wind directions and wind speeds (by index)
selected_wd_indices = [0, 1, 2, 3]   # e.g., 0°, 30°, 60°, 90°
selected_ws_indices = [0, 3, 6, 10]  # e.g., 4, 10, 16, 24 m/s

for i, (wd_idx, ws_idx) in enumerate(zip(selected_wd_indices, selected_ws_indices), 1):
    # Subset the yaw angles
    yaw_subset = state['yaw_ilk'][subset_indices, wd_idx, ws_idx]

    # Run simulation for just the subset
    sim_result = wf_model(
        x_subset, y_subset,
        wd=wd[wd_idx],
        ws=ws[ws_idx],
        yaw=yaw_subset,
        tilt=0
    )

    # Plot
    plt.figure(figsize=(12, 4))
    sim_result.flow_map().plot_wake_map()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f"Wake Map (10 turbines) at wd={wd[wd_idx]}°, ws={ws[ws_idx]} m/s")
    plt.tight_layout()
    plt.show()









# --- Optional: Plot AEP convergence ---
import matplotlib.pyplot as plt
plt.plot(aep_history, marker='o')
plt.title("AEP Progression During Optimization")
plt.xlabel("Iteration")
plt.ylabel("AEP (GWh)")
plt.grid(True)
plt.tight_layout()
plt.show()













from scipy.interpolate import make_interp_spline

# Ensure your AEP history is a NumPy array
aep_array = np.array(aep_history)
iterations = np.arange(len(aep_array))

# Create a smoothed x-axis with 1000 points, scaled between real iteration values
x_new = np.linspace(iterations.min(), iterations.max(), 1000)

# Create a cubic spline interpolation of the AEP values
spl = make_interp_spline(iterations, aep_array, k=3)
aep_smooth = spl(x_new)

# Plot
plt.figure(figsize=(6, 5))
plt.plot(x_new, aep_smooth, color='blue', label='Smoothed AEP')
plt.scatter(iterations, aep_array, s=10, color='black', alpha=0.5, label='Actual Points')
plt.title("AEP Progression During Optimization")
plt.xlabel("Iteration")
plt.ylabel("AEP (GWh)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
