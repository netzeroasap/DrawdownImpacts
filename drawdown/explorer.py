import os
from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from pathlib import Path
_DATA_DIR = Path(__file__).parent.parent / "data"

## Read in standard GWP calculations- will need to update when F gases come in
gwp_df = pd.read_excel(_DATA_DIR/ "emissions/solutions/GWP_conversions.xlsx",header = 1)
GWP20={}
GWP20["CH4"]=gwp_df[gwp_df.iloc[:,0]=="CH4"]["GWP20 (CO2-eq)"].values[0]
GWP20["N2O"]=gwp_df[gwp_df.iloc[:,0]=="N2O"]["GWP20 (CO2-eq)"].values[0]
GWP20["CO2"]=1.

def step_function(timepoints, t_start,L):
    """
    Step function for technology adoption.

    Parameters:
        timepoints:   Array of year timepoints.
        t_start:      Year adoption begins (curve is 0 before this).
        L:            Saturation level (maximum adoption)

    """
    return np.where(timepoints >= t_start, L, 0.0)

def pulse(timepoints, t_start, L):
    """
    Pulse function for instantaneous release

    Parameters:
        timepoints:   Array of year timepoints.
        t_start:      Year of implementation (curve is 0 before AND after this).
        L:            Saturation level (maximum adoption)

    """
    
    return np.where(timepoints == t_start, L, 0.0)

def rectangular_pulse(timepoints, t_start, L,duration=30):
    """
    Rectangular pulse function for constant release over time frame

    Parameters:
        timepoints:   Array of year timepoints.
        t_start:      Year of implementation (curve is 0 before this).
        duration:     How long (in years) the pulse lasts
        L:            Saturation level (maximum adoption)

    """
    t_stop=t_start+duration
    condition = np.logical_and(timepoints>t_start,timepoints<t_stop)
    return np.where(condition,L/duration,0.0)
    
    return np.where(timepoints == t_start, L, 0.0)
# def rectangular_pulse(timepoints,t_start,duration,L):
#      """
#     Pulse function for constant release over time frame

#     Parameters:
#         timepoints:   Array of year timepoints.
#         t_start:      Year of implementation (curve is 0 before this).
#         duration:     Length of pulse
#         L:            Saturation level (maximum adoption)

#     """
    
#     return(np.where(np.logical_and(timepoints >=t_start,timepoints<tstart+duration)),L/duration,0)

def S_curve(timepoints, t_start, L, k=0.5, years_to_half_adoption=None):
    """
    Logistic S-curve for technology adoption.

    Parameters:
        timepoints:   Array of year timepoints.
        t_start:      Year adoption begins (curve is 0 before this).
        L:            Saturation level (maximum adoption).
        k:            Growth rate / steepness. 
                        ~0.1 = slow, ~0.5 = moderate, ~1.0+ = fast S-curve.
        years_to_half_adoption: Years after t_start at which adoption is at 50% of L.
                        Defaults to 1/4 of the remaining time window.
    """
    t_end = timepoints[-1]
    if years_to_half_adoption is None:
        years_to_half_adoption = (t_end - t_start) / 4
    
    t_mid = t_start + years_to_half_adoption  # inflection point in absolute years

    raw = L / (1 + np.exp(-k * (timepoints - t_mid)))
    offset = L / (1 + np.exp(-k * (t_start - t_mid)))
    scale = L / (1 + np.exp(-k * (t_end - t_mid)))  - offset
    scaled = (raw - offset) / scale * L

    return np.where(timepoints < t_start, 0.0, np.clip(scaled, 0.0, L))

def linear_ramp(timepoints, t_start, L):
    """
    Linear ramp-up for technology adoption.

    Parameters:
        timepoints:   Array of year timepoints.
        t_start:      Year adoption begins (curve is 0 before this).
        L:            Saturation level (maximum adoption)
    """
    t_end =timepoints[-1]
    slope = L / (t_end - t_start)
    return np.where(timepoints < t_start, 0.0, slope * (timepoints - t_start))

def get_input(solution,\
              adoption_curve,\
              units_adopted,\
              implementation_start,\
              land=False,\
              function_kwargs={},\
             scenario_name=None):
    fair_start_time = 1750
    fair_end_time = 2101
    timepoints = np.arange(fair_start_time,fair_end_time+1,1)          
    #ALL OF THESE ARE IN CO2-eq-20/adoption unit/yr
    emissions_df=pd.read_csv(_DATA_DIR/ "emissions/solutions/reduced_emissions.csv")
    # if (len(np.where([x.find(solution)>=0 for x in emissions_df["solution name"]])[0]) and subcategory is None):
    #     solution += " Global weighted average"
    if solution not in emissions_df["solution name"].values:
        raise TypeError("Solution not found")
    else:
        subset=emissions_df[emissions_df["solution name"] == solution]
    perturbation_dict={}
    for specie in ["CH4","N2O","CO2"]:
        
        emissions=subset[specie].values[0]
        if emissions != 0:
            gwp20=GWP20[specie]
            if specie == "CO2":
                units_for_fair=1.e-9
                if land:
                    specie = "CO2 AFOLU"
                else:
                    specie = "CO2 FFI"
            else:
                units_for_fair=1.e-6
            L=units_adopted*emissions/gwp20*units_for_fair
            target = -1 * L
            sign = -1.0 if target < 0 else 1.0
            perturbation_dict[specie]= sign*adoption_curve(timepoints,\
                                                      implementation_start,\
                                                     sign*target,\
                                                      **function_kwargs)
    if scenario_name is None:
        scenario_name=solution
    return {scenario_name:perturbation_dict}


def drawdown_model(baseline_scenario, perturbation_dicts, df_configs=None,
                   start=1750, end=2101, lite=False, \
                   n_lite=10, emissions_year = 2024, random_seed=42, run=True):
    """
    Build and run a FaIR model instance

    Parameters:
        baseline_scenario: what emissions would do without action
        perturbation_dicts: dictionary of form d[scenario][specie]=timeseries of avoided emissions
        df_configs: parameters for the box model parameters 
        (ocean heat uptake, feedbacks, etc.)
        defaults to '../data/fair-parameters/calibrated_constrained_parameters_1.4.1.csv'
        start: start date (keep at 1750 for spinup)
        end: end date of simulation
        lite: run a configuration in reduced parameter space (saves memory)
        n_lite: the number of parameter values to use in lite mode
        emissions_year: when using a constant baseline, the year in which to freeze emissions
    """
    
    all_scenarios = [baseline_scenario] + list(perturbation_dicts.keys())

    
    if df_configs is None:
        df_configs = pd.read_csv(
            _DATA_DIR / 'fair-parameters/calibrated_constrained_parameters_1.4.1.csv',
            index_col=0
        )
    # ---lite mode: just pick n_lite configurations ---
    if lite:
        df_configs = df_configs.sample(n=n_lite, random_state=random_seed)
    
    species, properties = read_properties(
        _DATA_DIR / 'fair-parameters/species_configs_properties_1.4.1.csv'
    )
    # --- temporary instance just to load the CSV ---
    f_tmp = FAIR()
    f_tmp.define_time(start, end+1, 1)
    if baseline_scenario != "constant":
        f_tmp.define_scenarios([baseline_scenario])
    else:
        f_tmp.define_scenarios(["high-extension"])
    f_tmp.define_species(species, properties)
    f_tmp.define_configs(df_configs.index)
    f_tmp.allocate()
    f_tmp.fill_from_csv(
        emissions_file=_DATA_DIR / 'emissions/extensions_1750-2500.csv',
        forcing_file=_DATA_DIR / 'forcing/volcanic_solar.csv',
    )
    timepoints = f_tmp.timepoints
    timebounds = f_tmp.timebounds
    
    if baseline_scenario != "constant":
        baseline_emissions = f_tmp.emissions.sel(scenario=baseline_scenario).copy()
        baseline_forcing = f_tmp.forcing.sel(scenario=baseline_scenario).copy()
    else:
        high_emissions = f_tmp.emissions.sel(scenario="high-extension").copy()
        
        high_forcing = f_tmp.forcing.sel(scenario="high-extension").copy()
        #### DEFINE CONSTANT EMISSIONS #####
        emissions_const=high_emissions.sel(timepoints=emissions_year,method="nearest")
        
        #step = step_function(timepoints, emissions_year)[:, np.newaxis, np.newaxis]
        step = np.where(timepoints<emissions_year,0.0,1.0)[:, np.newaxis, np.newaxis]
        baseline_emissions = (                                                        
          high_emissions.values * (1 - step) +       # ramps to 0 at 2024
          emissions_const.values[np.newaxis, :, :] * step  # ramps on at 2024        
        )
        
                                                                                                                                                    
        forcing_const = high_forcing.sel(timebounds=emissions_year, method="nearest")                                                                
                                                                                    
        step_bounds = np.where(timebounds<emissions_year,0.0,1.0 )[:, np.newaxis, np.newaxis] 
                                                                                    
        baseline_forcing = (
          high_forcing.values * (1 - step_bounds) +
          forcing_const.values[np.newaxis, :, :] * step_bounds
        )    
       
    del f_tmp

    # --- real instance with all scenarios ---
    f = FAIR()
    f.define_time(start, end+1, 1)
    f.define_scenarios(all_scenarios)
    f.define_species(species, properties)
    f.ch4_method = 'Thornhill2021'
    f.define_configs(df_configs.index)
    f.allocate()
    f.fill_species_configs(
        _DATA_DIR / 'fair-parameters/species_configs_properties_1.4.1.csv'
    )
    f.override_defaults(
        _DATA_DIR / 'fair-parameters/calibrated_constrained_parameters_1.4.1.csv'
    )
   
    for s in all_scenarios:
        f.emissions.loc[dict(scenario=s)] = baseline_emissions
        f.forcing.loc[dict(scenario=s)] = baseline_forcing

    for s, perturbation in perturbation_dicts.items():
        for specie, delta in perturbation.items():
            f.emissions.loc[dict(scenario=s, specie=specie)] += delta[:, np.newaxis]

    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)
    
    f.run()
    return f


def plot_T_diff(f,baseline_scenario="constant",end_time=2100,scenarios=None):
    baseline_T=f.temperature.sel(layer=0,\
                    scenario=baseline_scenario,\
                   timebounds=slice(2020,end_time))
    if scenarios is None:
        scenarios=f.scenarios
    for scenario in scenarios:
        if scenario != baseline_scenario:
            scen_T=f.temperature.sel(layer=0,\
                    scenario=scenario,\
                   timebounds=slice(2020,end_time))
            deltaT=scen_T-baseline_T
            tax=deltaT.timebounds.values
            median = deltaT.median(dim="config")
            lo = deltaT.quantile(0.25, dim="config")
            hi = deltaT.quantile(0.75, dim="config")
        
            plt.plot(tax, median, label=scenario)
            plt.fill_between(tax, lo, hi, alpha=0.2)
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Warming avoided (°C)")