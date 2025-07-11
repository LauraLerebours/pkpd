import numpy as np
from scipy.integrate import odeint # For solving ordinary differential equations
import matplotlib.pyplot as plt

# --- 1. Model Parameters (Extracted from Source 5, Tables 1 & 2) ---
# These parameters are based on age and typically vary with lean body mass (LBM).
# The provided tables [4] give discrete values for specific age groups (20, 50, 80 years).
# LBM 55 kg is used as a representative value where not explicitly differentiated by LBM in tables.
# Note: For real applications, covariate functions (e.g., Cl = Cl_ref * (Age/70)^a * (LBM/55)^b)
# would be derived during model estimation, but these explicit continuous functions
# are not provided in the excerpts. We use a lookup approach for simplicity in this simulation example.

MODEL_PARAMS_BY_COVARIATES = {
    # Key: (Age, LBM) - LBM is used for consistency, but the provided tables
    # primarily show age dependence directly for most parameters without LBM as a direct multiplier.
    # Cl values are from Table 2 [4], other PK/PD parameters are from Table 1 [4].
    # (Note: For 50yr, Table 2 [4] shows identical Cl for 35, 55, and 75kg LBM;
    # however, Figure 2 nomograms [5] show clear LBM influence on infusion rates,
    # implying more complex LBM relationships in the full model not fully detailed in tables).
    (20, 55): {
        'Cl': 2.9,     # L/min (Clearance) [5, Table 2]
        'V1': 5.5,     # L (Central Volume) [5, Table 1]
        'k_e0': 0.94,  # min^-1 (Effect-site equilibration rate constant) [5, Table 1]
        'EC50': 17.35, # ng/mL (Effect-site concentration for 50% maximum effect) [5, Table 1]
        'Vdss': 17.3,  # L (Volume of distribution at steady state) [5, Table 1]
        'E0_EEG': 100, # Baseline EEG effect (e.g., 100% of normal brain activity)
        'Emax_EEG': 100, # Maximum possible reduction from baseline (e.g., to 0%)
        'Gamma_EEG': 1 # Hill coefficient, assumed 1 (simple Emax) as not specified in source excerpts.
    },
    (50, 55): {
        'Cl': 2.7,
        'V1': 5.1,
        'k_e0': 1.32,
        'EC50': 16.97,
        'Vdss': 17.0,
        'E0_EEG': 100,
        'Emax_EEG': 100,
        'Gamma_EEG': 1
    },
    (80, 55): {
        'Cl': 2.0,
        'V1': 4.3,
        'k_e0': 2.20,
        'EC50': 17.30,
        'Vdss': 16.2,
        'E0_EEG': 100,
        'Emax_EEG': 100,
        'Gamma_EEG': 1
    },
    # Adding other LBMs as per Table 2 [4] for 50-year-olds.
    # Note: Cl values are identical for different LBMs at 50yr in this specific table,
    # meaning the influence on Cl alone is not seen here, but is implied in nomograms [5].
    (50, 35): {
        'Cl': 2.7, 'V1': 5.1, 'k_e0': 1.32, 'EC50': 16.97, 'Vdss': 17.0,
        'E0_EEG': 100, 'Emax_EEG': 100, 'Gamma_EEG': 1
    },
    (50, 75): {
        'Cl': 2.7, 'V1': 5.1, 'k_e0': 1.32, 'EC50': 16.97, 'Vdss': 17.0,
        'E0_EEG': 100, 'Emax_EEG': 100, 'Gamma_EEG': 1
    }
}


class RemifentanilPKPDModel:
    """
    A simplified Pharmacokinetic (PK) and Pharmacodynamic (PD) model for Remifentanil.

    NOTE: This code implements a *simulation* of PK/PD profiles based on
    published parameters. It does NOT perform parameter *estimation* from raw data,
    which typically requires specialized software like NONMEM [2, from conversation history].

    Simplifications made due to incomplete explicit mathematical formulas in the provided sources:
    - The full multi-compartment PK model differential equations for remifentanil
      are not explicitly provided. Plasma concentration (Cp) is approximated using
      a simplified one-compartment elimination model driven by infusion rate and Cl.
      A true complex model would involve more detailed compartmental dynamics [2].
    - Hill coefficient (Gamma_EEG) for the sigmoidal Emax model is assumed to be 1,
      as it is not explicitly stated in the provided excerpts for remifentanil.
    - Covariate (Age, LBM) effects on parameters are handled via direct look-up of
      tabulated values [4], rather than continuous functions.
    """

    def __init__(self, age: int, lbm: int = 55):
        """
        Initializes the PK/PD model with parameters for a specific patient.

        Args:
            age (int): Patient's age in years (e.g., 20, 50, 80). [1]
            lbm (int): Patient's Lean Body Mass in kg (e.g., 35, 55, 75). [2]
        """
        # Select parameters based on the closest available age/LBM in the lookup table.
        # In a real, full model, continuous covariate functions would be used.
        patient_params_key = (age, lbm)
        if patient_params_key not in MODEL_PARAMS_BY_COVARIATES:
            raise ValueError(f"Parameters for Age {age} and LBM {lbm} not explicitly defined in lookup table from sources. Supported combinations: {list(MODEL_PARAMS_BY_COVARIATES.keys())}")

        self.params = MODEL_PARAMS_BY_COVARIATES[patient_params_key]
        self.age = age
        self.lbm = lbm

        # Extract specific parameters for easier access
        self.Cl = self.params['Cl']         # Clearance (L/min) [5, Table 2]
        self.V1 = self.params['V1']         # Central Volume (L) [5, Table 1]
        self.k_e0 = self.params['k_e0']     # Effect-site equilibration rate constant (min^-1) [5, Table 1]
        self.EC50 = self.params['EC50']     # Effect-site concentration for 50% max effect (ng/mL) [5, Table 1]
        self.E0_EEG = self.params['E0_EEG'] # Baseline EEG effect (e.g., 100%)
        self.Emax_EEG = self.params['Emax_EEG'] # Max possible EEG effect (e.g., 100% reduction)
        self.Gamma_EEG = self.params['Gamma_EEG'] # Hill coefficient (assumed 1)

        print(f"Model initialized for Age: {self.age} yrs, LBM: {self.lbm} kg with parameters:")
        for k, v in self.params.items():
            print(f"  {k}: {v}")


    def _pk_ode_system(self, y, t, infusion_rate_func):
        """
        Defines the ordinary differential equations (ODEs) for the PK part.
        y = Plasma Concentration (Cp)
        y[1] = Effect Site Concentration (Ce)

        NOTE: This is a highly simplified PK model (1-compartment for Cp, then effect site).
        A full remifentanil model is multi-compartmental [2].
        """
        Cp, Ce = y

        # Infusion rate at time t (in ng/min)
        infusion_rate = infusion_rate_func(t)

        # Change in plasma concentration (dCp/dt)
        # Assuming a one-compartment model for simplicity.
        # Units: infusion_rate (ng/min), V1 (L), Cl (L/min), Cp (ng/mL)
        # Convert V1 to mL for unit consistency: V1 * 1000 mL/L
        dCp_dt = (infusion_rate / (self.V1 * 1000)) - (self.Cl / self.V1) * Cp

        # Change in effect site concentration (dCe/dt) [1]
        dCe_dt = self.k_e0 * (Cp - Ce)

        return [dCp_dt, dCe_dt]


    def simulate(self,
                 total_time: float,
                 time_points: np.ndarray,
                 initial_bolus_mg: float = 0.0,
                 infusion_rate_profile: callable = lambda t: 0.0):
        """
        Simulates the PK and PD profile over time.

        Args:
            total_time (float): Total simulation time in minutes.
            time_points (np.ndarray): Array of time points (in minutes) at which to report results.
            initial_bolus_mg (float): Initial bolus dose in mg. Will be converted to ng.
                                      (e.g., 200 µg = 0.2 mg). Source [5] mentions 200 µg bolus.
            infusion_rate_profile (callable): A function `f(t)` that returns the
                                              infusion rate in ng/min at time `t`.
                                              (e.g., 100 µg/min = 100,000 ng/min).

        Returns:
            tuple: (time_points, Cp_history, Ce_history, EEG_effect_history)
        """
        # Convert initial bolus from mg to ng for consistency with other units (EC50 in ng/mL)
        initial_bolus_ng = initial_bolus_mg * 1_000_000 # 1 mg = 1,000,000 ng

        # Initial Plasma Concentration (Cp) from bolus (simplified: instantaneous distribution in V1)
        initial_Cp = initial_bolus_ng / (self.V1 * 1000) if initial_bolus_ng > 0 else 0.0
        initial_Ce = 0.0 # Start with no drug at effect site

        y0 = [initial_Cp, initial_Ce]

        # Solve the ODE system using scipy's odeint
        solution = odeint(self._pk_ode_system, y0, time_points,
                          args=(infusion_rate_profile,)) # Pass the infusion rate function

        Cp_history = solution[:, 0]
        Ce_history = solution[:, 1]

        # Calculate PD effect using the sigmoidal Emax model [from conversation history]
        # Effect = E0 - Emax * (Ce^Gamma / (EC50^Gamma + Ce^Gamma))
        # Since EEG effect is described as "reduction" [6], E0 is baseline (100%),
        # and Emax is the maximum possible reduction (100%), so effect decreases from E0.
        EEG_effect_history = self.E0_EEG - (self.Emax_EEG * (Ce_history**self.Gamma_EEG) /
                                            (self.EC50**self.Gamma_EEG + Ce_history**self.Gamma_EEG))

        # Ensure EEG effect does not go below 0%
        EEG_effect_history = np.maximum(0, EEG_effect_history)

        return time_points, Cp_history, Ce_history, EEG_effect_history

    def plot_results(self, time_points, Cp_history, Ce_history, EEG_effect_history, scenario_title=""):
        """Plots the simulated PK and PD profiles."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plasma concentration
        axes[0].plot(time_points, Cp_history, label='Plasma Concentration (Cp)')
        axes[0].set_ylabel('Concentration (ng/mL)')
        axes[0].set_title(f'Remifentanil PK/PD Simulation ({scenario_title})\nAge: {self.age} yrs, LBM: {self.lbm} kg')
        axes[0].legend()
        axes[0].grid(True)

        # Effect site concentration
        axes[1].plot(time_points, Ce_history, label='Effect Site Concentration (Ce)', color='orange')
        axes[1].set_ylabel('Concentration (ng/mL)')
        axes[1].legend()
        axes[1].grid(True)

        # EEG effect
        axes[2].plot(time_points, EEG_effect_history, label='EEG Effect (% of Baseline)', color='green')
        axes[2].set_ylabel('EEG Effect (%)')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].axhline(y=self.E0_EEG - (self.Emax_EEG * 0.5), color='red', linestyle='--', label='50% Reduction Level')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        


# --- Example Usage Scenarios ---
if __name__ == "__main__":
    # Define common simulation time points
    sim_time = 60 # minutes
    n_points = 500
    times = np.linspace(0, sim_time, n_points)

    print("---------------------------------------------------------")
    print("      Remifentanil PK/PD Model Simulations              ")
    print("  (Based on simplified models and parameters from sources)")
    print("---------------------------------------------------------\n")

    # --- Scenario 1: Typical Adult (50-year-old, 55kg LBM) ---
    print("\n--- Scenario 1: 50-year-old, 55kg LBM, Bolus + Infusion ---")
    patient_model_50yo = RemifentanilPKPDModel(age=50, lbm=55)

    # Example infusion rate to achieve a significant effect.
    # Figure 2 [5] suggests ~20 µg/min for 50yr, 55kg LBM for 50% EEG effect.
    typical_infusion_rate_ng_per_min = 20_000 # ng/min (20 µg/min)

    def infusion_func_50yo(t):
        if 0 <= t <= sim_time: # Infuse for the whole duration
            return typical_infusion_rate_ng_per_min
        return 0.0

    # Simulate with an initial bolus (e.g., 200 µg = 0.2 mg) [5]
    t_50, Cp_50, Ce_50, EEG_50 = patient_model_50yo.simulate(
        total_time=sim_time,
        time_points=times,
        initial_bolus_mg=0.2, # mg (200 µg)
        infusion_rate_profile=infusion_func_50yo
    )
    patient_model_50yo.plot_results(t_50, Cp_50, Ce_50, EEG_50, scenario_title="50-Year-Old, 55kg LBM")


    # --- Scenario 2: Elderly Patient (80-year-old, 55kg LBM) ---
    # The sources highlight age-dependent changes in PK and PD, with reduced clearance [3, 4]
    # and faster k_e0 in elderly [4].
    print("\n--- Scenario 2: 80-year-old, 55kg LBM, Bolus + Infusion ---")
    patient_model_80yo = RemifentanilPKPDModel(age=80, lbm=55)

    # Adjust infusion rate based on nomogram for 80-year-old for 50% EEG effect.
    # Figure 2 [5] suggests ~10 µg/min for 80yr, 55kg LBM.
    typical_infusion_rate_80yo_ng_per_min = 10_000 # ng/min (10 µg/min)

    def infusion_func_80yo(t):
        if 0 <= t <= sim_time:
            return typical_infusion_rate_80yo_ng_per_min
        return 0.0

    # Use an age-adjusted bolus as well. Figure 2 [5] suggests ~100 µg for 80yr.
    t_80, Cp_80, Ce_80, EEG_80 = patient_model_80yo.simulate(
        total_time=sim_time,
        time_points=times,
        initial_bolus_mg=0.1, # mg (100 µg)
        infusion_rate_profile=infusion_func_80yo
    )
    patient_model_80yo.plot_results(t_80, Cp_80, Ce_80, EEG_80, scenario_title="80-Year-Old, 55kg LBM")

    # --- Scenario 3: Patient with Different LBM (50-year-old, 35kg LBM) ---
    # While Table 2 [4] shows identical Cl for 50yr across LBMs, Figure 2 nomograms [5]
    # clearly show LBM dependency for infusion rate and bolus dose. This implies the full model
    # has more nuanced LBM covariate effects (e.g., on Vd) than just what's in Table 2 for Cl.
    # We will use the Cl from Table 2 and the same k_e0/EC50 from 55kg, but use dose from Nomogram.
    print("\n--- Scenario 3: 50-year-old, 35kg LBM ---")
    patient_model_50yo_35lbm = RemifentanilPKPDModel(age=50, lbm=35)
    # Infusion rate from Figure 2 [5] for 50yr, 35kg LBM is lower, ~15 µg/min
    infusion_35lbm_ng_per_min = 15_000 # ng/min (15 µg/min)
    t_50_35, Cp_50_35, Ce_50_35, EEG_50_35 = patient_model_50yo_35lbm.simulate(
        total_time=sim_time,
        time_points=times,
        initial_bolus_mg=0.15, # mg (150 µg for 50yr, 35kg from Figure 2 [5])
        infusion_rate_profile=lambda t: infusion_35lbm_ng_per_min if 0 <= t <= sim_time else 0.0
    )
    patient_model_50yo_35lbm.plot_results(t_50_35, Cp_50_35, Ce_50_35, EEG_50_35, scenario_title="50-Year-Old, 35kg LBM")
    plt.show()
    print("\n---------------------------------------------------------")
    print("IMPORTANT NOTE ON LBM INFLUENCE IN THESE SIMULATIONS:")
    print("While the model structure includes LBM as a covariate, the specific excerpted tables [4] ")
    print("only show age-dependent changes for most PK/PD parameters (V1, k_e0, EC50, Vdss).")
    print("Table 2 [4] *does* list Cl for different LBMs, but shows identical Cl for 50yr at 35, 55, 75kg LBMs.")
    print("However, Figure 2 (Nomograms) [5] clearly illustrates LBM-dependent bolus doses and infusion rates.")
    print("This implies the full model has more nuanced LBM covariate effects (e.g., on Vd) than")
    print("what is explicitly detailed in the provided parameter tables for the chosen LBMs.")
    print("Therefore, in this Python simulation, unless the tables provide different values,")
    print("the PK/PD behavior for varying LBMs (like in Scenario 3 vs Scenario 1) might not fully reflect ")
    print("the LBM dependency seen in the nomograms, but rather only what is directly parameterized from the tables.")
    print("A complete model from the source's supplementary materials would provide these details.")
    print("---------------------------------------------------------")