#model based on paper by Steven L Shafer MD
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
# PK STUFF
def triexponential_model(t, A, alpha, B, beta, C, gamma):
    """
    Triexponential pharmacokinetic model function.
    t: Time
    A, B, C: Coefficients representing volume terms (A1, A2, A3)
    alpha, beta, gamma: Rate constants
    """
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t) + C * np.exp(-gamma * t)

# example concentration data
# In a real study, this would be your observed plasma drug concentration data.
# We'll generate data that broadly resembles typical PK curves for demonstration.
time_points = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 12, 18, 24]) # Example time points in hours or minutes [1]

# True (but unknown in a real scenario) parameters for simulation
# These are chosen to generate a curve that can be fitted.
# They are NOT the actual parameters from the study.
true_A = 6.09   # Corresponds to A1
true_alpha = 10 # Rapid distribution phase rate
true_B = 0.504  # Corresponds to A2
true_beta = 1   # Intermediate distribution phase rate
true_C = 0.0100 # Corresponds to A3
true_gamma = 0.1 # Slow elimination phase rate


# Generate "observed" concentrations with some added noise to simulate real data
observed_concentrations = triexponential_model(time_points, true_A, true_alpha, true_B, true_beta, true_C, true_gamma)
# Add random noise to simulate measurement error
np.random.seed(42) # for reproducibility
noise = np.random.normal(0, 0.01, observed_concentrations.shape) # Small Gaussian noise
observed_concentrations = observed_concentrations + noise

# Ensure no negative concentrations due to noise, which wouldn't be physical
observed_concentrations[observed_concentrations < 0] = 0.001

# Initial guesses for the parameters are important for the optimization algorithm.
# These guesses should be reasonable based on the expected shape of the curve.
# If good initial guesses are not known, broad ranges can be tried, or visual inspection of data.
# The parameters are [A, alpha, B, beta, C, gamma]
initial_guesses = [5, 15, 0.5, 2, 0.01, 0.05] # Example initial guesses

# Bounds can help guide the optimization to physiologically plausible ranges
# For example, coefficients (A, B, C) should be positive, rates (alpha, beta, gamma) should be positive.
# Rates should also generally follow alpha > beta > gamma.
bounds = ([0 for _ in range(6)], [np.inf for _ in range(6)])
A = []
try:
    # curve_fit returns:
    # popt: Optimal values for the parameters so that the sum of the squared residuals of `f(xdata, *popt) - ydata` is minimized.
    # pcov: The estimated covariance of popt.
    popt, pcov = curve_fit(triexponential_model, time_points, observed_concentrations, p0=initial_guesses, bounds=bounds)

    # Extract the fitted parameters [3]
    fitted_A, fitted_alpha, fitted_B, fitted_beta, fitted_C, fitted_gamma = popt

    # A1, A2, A3 correspond to A, B, C from the fitted model [4, 5]
    fitted_A1 = fitted_A
    fitted_A2 = fitted_B
    fitted_A3 = fitted_C

    A = [fitted_A1,fitted_A2, fitted_A3] # to use later

    print("--- Fitted Pharmacokinetic Parameters ---")
    print(f"A1 (fitted A): {fitted_A1:.4f} L/kg")
    print(f"alpha (fitted alpha): {fitted_alpha:.4f}")
    print(f"A2 (fitted B): {fitted_A2:.4f} L/kg")
    print(f"beta (fitted beta): {fitted_beta:.4f}")
    print(f"A3 (fitted C): {fitted_A3:.4f} L/kg")
    print(f"gamma (fitted gamma): {fitted_gamma:.4f}")

    # Visualize the Fit
    plt.figure(figsize=(10, 6))
    plt.scatter(time_points, observed_concentrations, label='Observed Concentrations (Simulated)', color='blue')
    plt.plot(time_points, triexponential_model(time_points, *popt), label='Fitted Triexponential Model', color='red')
    plt.xscale('log') # Often concentration-time data is plotted on a log scale for time
    plt.xlabel('Time (e.g., minutes/hours)')
    plt.ylabel('Concentration (e.g., ng/mL)')
    plt.title('Simulated Fentanyl Pharmacokinetics and Model Fit')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')

except RuntimeError as e:
    print(f"Error: Could not fit curve. This can happen if initial guesses are poor or data is insufficient/noisy. Details: {e}")


# PD STUFF


# Fentanyl Pharmacokinetic Parameters from Shafer et al. (1990) - "Current Study" 

# Volumes (L) 
V1 = 6.09   # Central Compartment Volume
V2 = 22.8   # Rapid Peripheral Compartment Volume
V3 = 228    # Slow Peripheral Compartment Volume

# Rate Constants (per minute) 
# Note: k_ij means transfer from compartment i to compartment j
k10 = 0.0827  # Elimination from Central (1 -> 0)
k12 = 0.294   # Central to Rapid Peripheral (1 -> 2)
k21 = 0.131   # Rapid Peripheral to Central (2 -> 1)
k13 = 0.00800 # Central to Slow Peripheral (1 -> 3)
k31 = 0.000840 # Slow Peripheral to Central (3 -> 1)

# Clearances (L/min) - can be derived from V and k, or specified directly.
# The paper provides clearances in Table 1, but uses rate constants and volumes in the ODEs.
# For example, CL = k10 * V1 is the elimination clearance from the central compartment.
CL12 = k12 * V1 # Clearance from central to rapid peripheral
CL13 = k13 * V1 # Clearance from central to slow peripheral
CL21 = k21 * V2 # Clearance from rapid peripheral to central
CL31 = k31 * V3 # Clearance from slow peripheral to central
CL_elim = k10 * V1 # Elimination clearance from central

def get_current_infusion_rate(t):
    if t <= 60: # Infuse for the first 60 minutes
        return infusion_rate_mg_per_min
    else:
        return 0

# Modify the ODE function to take time-varying infusion rate
def fentanyl_pk_model_time_varying(A, t, params):
    """
    Defines the system of ordinary differential equations for the
    three-compartment mammillary pharmacokinetic model of fentanyl.

    Args:
        A (list): Current amounts of drug in each compartment [A1, A2, A3].
                  A1: Central compartment (e.g., blood)
                  A2: Rapid peripheral compartment
                  A3: Slow peripheral compartment
        t (float): Current time.
        infusion_rate (float): The rate of drug infusion into the central compartment (e.g., mg/min or mcg/min).
        params (dict): Dictionary containing the pharmacokinetic parameters (rate constants).

    Returns:
        list: Rates of change of drug amount in each compartment [dA1/dt, dA2/dt, dA3/dt].
    """
    
    A1, A2, A3 = A

    k10 = params['k10']
    k12 = params['k12']
    k21 = params['k21']
    k13 = params['k13']
    k31 = params['k31']

    current_infusion = get_current_infusion_rate(t) # Get the infusion rate at the current time

    dA1dt = current_infusion - (k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3
    dA2dt = k12 * A1 - k21 * A2
    dA3dt = k13 * A1 - k31 * A3

    return [dA1dt, dA2dt, dA3dt]


def get_weight_scaled_params(patient_weight_kg):
    """
    Returns weight-scaled pharmacokinetic parameters for fentanyl.
    Parameters are from Shafer et al. (1990) Table 1, "Weight-scaled" column [3].
    Note: The original paper provides scaled *volumes* and *clearances*.
    We convert these to volumes and rate constants for the ODEs.

    Args:
        patient_weight_kg (float): The patient's weight in kilograms.

    Returns:
        dict: A dictionary of weight-scaled pharmacokinetic parameters.
    """
    # Weight-scaled volumes (L/kg) from Table 1 [3]
    V1_per_kg = 0.087  # Central volume
    V2_per_kg = 0.326  # Rapid peripheral volume
    V3_per_kg = 3.26   # Slow peripheral volume

    V1_scaled = V1_per_kg * patient_weight_kg
    V2_scaled = V2_per_kg * patient_weight_kg
    V3_scaled = V3_per_kg * patient_weight_kg

    # Weight-scaled clearances (L/min/kg) from Table 1 [3]
    CL_elim_per_kg = 0.00504 # Elimination clearance (from central)
    CL12_per_kg = 0.0232   # Clearance from central to rapid
    CL13_per_kg = 0.000730 # Clearance from central to slow

    k10_scaled = CL_elim_per_kg * patient_weight_kg / V1_scaled # CL_elim / V1
    k12_scaled = CL12_per_kg * patient_weight_kg / V1_scaled # CL12 / V1
    k13_scaled = CL13_per_kg * patient_weight_kg / V1_scaled # CL13 / V1

    k21_scaled = 0.131 # Table 2, "Weight-scaled", k21 [2]
    k31_scaled = 0.00084 # Table 2, "Weight-scaled", k31 [2]


    return {
        'V1': V1_scaled, 'V2': V2_scaled, 'V3': V3_scaled,
        'k10': k10_scaled, 'k12': k12_scaled, 'k21': k21_scaled, # k21 and k31 are not weight-scaled in Table 2 [2]
        'k13': k13_scaled, 'k31': k31_scaled
    }

# Example of obtaining parameters
patient_weight_kg = 70
params_fixed = {'V1': V1, 'V2': V2, 'V3': V3, 'k10': k10, 'k12': k12, 'k21': k21, 'k13': k13, 'k31': k31}
params_scaled = get_weight_scaled_params(patient_weight_kg)


# 1. Choose parameters (fixed or weight-scaled)
current_params = params_fixed # Or params_scaled if using weight scaling

# 2. Set initial conditions: amounts of drug in each compartment (A1, A2, A3)
initial_amounts = tuple([a for a in A])  # Starting with no drug in the system [Previous response]

# 3. Define simulation time (e.g., 240 minutes = 4 hours)
time_points = np.linspace(0, 240, 500) # Simulate over 240 minutes, with 500 points

# 4. Define infusion schedule (e.g., a constant infusion rate)
# Let's say a constant infusion of 10 mcg/min (0.01 mg/min) for the first 60 minutes, then off.
# This simple setup might not directly reflect a CCIP's complex targeting, but demonstrates the model.
# For CCIP, the infusion rate would change over time based on target concentration feedback.
infusion_rate_mcg_per_min = 10 # Example infusion rate
infusion_rate_mg_per_min = infusion_rate_mcg_per_min / 1000 # Convert to mg if using mg for drug amount

solution = odeint(fentanyl_pk_model_time_varying, initial_amounts, time_points, args=(current_params,))

A1_over_time = solution[:, 0] # Amount in central compartment
A2_over_time = solution[:, 1] # Amount in rapid peripheral
A3_over_time = solution[:, 2] # Amount in slow peripheral

predicted_C1_mg_per_L = A1_over_time / current_params['V1']
predicted_C1_ng_per_mL = predicted_C1_mg_per_L * 1000 # Convert mg/L to ng/mL (1 mg = 1000 mcg, 1 L = 1000 mL, 1 mcg/mL = 1 ng/mL)

plt.figure(figsize=(10, 6))
plt.plot(time_points, predicted_C1_ng_per_mL)

plt.xlabel("Time (minutes)")
plt.ylabel("Fentanyl Concentration (ng/mL)")
plt.title("Predicted Fentanyl Plasma Concentration over Time")
plt.grid(True)
plt.show()  