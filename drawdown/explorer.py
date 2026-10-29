import os
import re
from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from functools import lru_cache
_DATA_DIR = Path(__file__).parent.parent / "data"

## Reduced emissions per adoption unit live in the "Drawdown Climate Reduced
## Emissions Numbers" Google Sheet, read via the PD service account (see README)
REDUCED_EMISSIONS_SHEET_KEY = "10TxDO4DeNPRRTTaGix13Gfa5Q8WoolKJBeOKFzm-HzY"
REDUCED_EMISSIONS_WORKSHEET_ID = 484980951   # the "reduced_emissions" tab
SERVICE_ACCOUNT_FILE = Path(__file__).parent.parent / "service_account.json"

@lru_cache(maxsize=1)
def _fetch_reduced_emissions():
    import gspread
    if SERVICE_ACCOUNT_FILE.exists():
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
    else:
        gc = gspread.service_account()   # falls back to ~/.config/gspread/service_account.json
    ws = gc.open_by_key(REDUCED_EMISSIONS_SHEET_KEY)\
           .get_worksheet_by_id(REDUCED_EMISSIONS_WORKSHEET_ID)
    # UNFORMATTED so numbers come back at full precision, not as displayed
    rows = ws.get_all_values(value_render_option="UNFORMATTED_VALUE")
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df.replace("", np.nan)

def read_reduced_emissions(refresh=False):
    """
    Reduced emissions table (one row per solution), read from the Google Sheet.
    Fetched once per session and cached; refresh=True re-reads the sheet.
    """
    if refresh:
        _fetch_reduced_emissions.cache_clear()
    return _fetch_reduced_emissions().copy()

## Read in standard GWP calculations- will need to update when F gases come in
gwp_df = pd.read_excel(_DATA_DIR/ "emissions/solutions/GWP_conversions.xlsx",header = 1)
GWP20={}
GWP20["CH4"]=gwp_df[gwp_df.iloc[:,0]=="CH4"]["GWP20 (CO2-eq)"].values[0]
GWP20["N2O"]=gwp_df[gwp_df.iloc[:,0]=="N2O"]["GWP20 (CO2-eq)"].values[0]
GWP20["CO2"]=1.

TIMEPOINTS = np.arange(1750, 2102)   # matches get_input / drawdown_model

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
    
def linear_ramp(timepoints, t_start, L, t_end=None):
    """
    Linear ramp-up for technology adoption.

    Parameters:
        timepoints:   Array of year timepoints.
        t_start:      Year adoption begins (curve is 0 before this).
        L:            Saturation level (maximum adoption).
        t_end:        Year saturation is reached (curve stays at L after this).
                      Defaults to the last timepoint.
    """
    timepoints = np.asarray(timepoints, dtype=float)
    if t_end is None:
        t_end = timepoints[-1]
    if t_end <= t_start:
        raise ValueError("t_end must be after t_start")

    slope = L / (t_end - t_start)
    return np.clip(slope * (timepoints - t_start), 0.0, L)


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
    #emissions_df=pd.read_csv(_DATA_DIR/ "emissions/solutions/reduced_emissions.csv")
    ##### NEW 9/15: FIX
    emissions_df=read_reduced_emissions()
    # Replaces all commas across all columns with an empty string
    emissions_df = emissions_df.replace(',', '', regex=True)
    ### END NEW 9/15
    
    # if (len(np.where([x.find(solution)>=0 for x in emissions_df["solution name"]])[0]) and subcategory is None):
    #     solution += " Global weighted average"
    # Matched after normalizing (lowercase, '&'->'and', strip punctuation) rather
    # than exact string equality, since reduced_emissions.csv and
    # drawdown_adoption_ranges.xlsx don't always agree on spelling for the same
    # solution (e.g. "Paper & Cardboard" vs "Paper and Cardboard") -- see
    # achievable_adoption, which does the same normalization on the other table.
    key=_normalize_solution_name(solution)
    normalized_names=emissions_df["solution name"].map(_normalize_solution_name)
    exact=emissions_df[normalized_names == key]
    if len(exact) == 1:
        subset=exact
    else:
        prefix=emissions_df[[key.startswith(n) or n.startswith(key) for n in normalized_names]]
        if len(prefix) == 1:
            subset=prefix
        elif len(prefix) > 1:
            raise ValueError(f"'{solution}' matches multiple rows in reduced_emissions.csv: "
                             f"{list(prefix['solution name'])}")
        else:
            raise TypeError(f"Solution not found: '{solution}' (checked exact and prefix match "
                            "after normalizing '&'/'and' and punctuation)")
    perturbation_dict={}
    for specie in ["CH4","N2O","CO2"]:
        ## NEW 9/15: Changed to float###
        #emissions=subset[specie].values[0]
        emissions=float(subset[specie].values[0])
        ### END NEW 9/15
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
                   n_lite=10, emissions_year = 2024, random_seed=42, run_model=True,\
                  action="add", CH4_lifetime_effects=True):
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
        CH4_lifetime_effects: if False, zeroes ch4_lifetime_chemical_sensitivity for
            NOx/CO/VOC, i.e. turns off the chemistry channel through which those species'
            emissions alter methane's atmospheric lifetime (see replacement_model, which
            this mirrors). Default True keeps FaIR's normal behavior.
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
          high_emissions.values * (1 - step) +       # ramps to 0 at emissions_year
          emissions_const.values[np.newaxis, :, :] * step  # ramps on at emissions_year       
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
    if not CH4_lifetime_effects:
        for specie in ["NOx","CO","VOC"]:
            f.species_configs["ch4_lifetime_chemical_sensitivity"]\
                .loc[dict(specie=specie)] =0.

    for s in all_scenarios:
        f.emissions.loc[dict(scenario=s)] = baseline_emissions
        f.forcing.loc[dict(scenario=s)] = baseline_forcing

    for s, perturbation in perturbation_dicts.items():
        for specie, delta in perturbation.items():
            if action=="add":
                f.emissions.loc[dict(scenario=s, specie=specie)] += delta[:, np.newaxis]
            elif action == "replace":
                f.emissions.loc[dict(scenario=s, specie=specie)] = delta[:, np.newaxis]
            else:
                raise TypeError("action must be add or replace")
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)
    setattr(f,"baseline_scenario",baseline_scenario)
    if run_model:
        f.run()
    return f


def plot_T_diff(f,end_time=2100,scenarios=None,baseline_scenario=None):
    if baseline_scenario is None:
        baseline_scenario=f.baseline_scenario
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

def shadeplot(da,dim="config",timedim=None,hdi_prob=.9,**plt_kwargs):
    """
    Plot the shaded interval
    """
    low=(1-hdi_prob)/2
    med=.5
    high=1-(1-hdi_prob)/2
    quantiles=da.quantile([low,med,high],dim=dim)
    if timedim is None:
        if len(da.dims)==2:
            timedim=np.array(da.dims)[np.array(da.dims)!="config"][0]
        else:
            raise TypeError("array must be 2d")
    lowq=quantiles.sel(quantile=low)
    highq=quantiles.sel(quantile=high)
    xax=getattr(lowq,timedim).values
    ax=plt.fill_between(xax,lowq.values,highq.values,**plt_kwargs)
    return ax
        


def replacement_model(baseline_scenario, 
                      replacement_dicts, 
                      df_configs=None,
                      start=1750, 
                      end=2101, 
                      lite=False, 
                      n_lite=10, 
                      emissions_year=2024,
                      random_seed=42, 
                      run_model=True,
                     CH4_lifetime_effects=True):
    """
    Build and run a FaIR model instance

    Parameters:
        baseline_scenario: assumed reference scenario
        perturbation_dicts: dictionary of form d[scenario][specie]=replacement_scenario_name. Replace emissions of [specie] in the baseline_scenario with 
        df_configs: parameters for the box model parameters 
        (ocean heat uptake, feedbacks, etc.)
        defaults to '../data/fair-parameters/calibrated_constrained_parameters_1.4.1.csv'
        start: start date (keep at 1750 for spinup)
        end: end date of simulation
        lite: run a configuration in reduced parameter space (saves memory)
        n_lite: the number of parameter values to use in lite mode
 
    """
    # ---get the scenario names from the emissions file---#
    emissions_file=_DATA_DIR / 'emissions/extensions_1750-2500.csv'
    df=pd.read_csv(emissions_file)
    scenarios=np.array(df.scenario.unique())
    del df
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
    # --- temporary instance just to load the CSV of high/low emissions ---
    
    f_tmp = FAIR()
    f_tmp.define_time(start, end+1, 1)
    f_tmp.define_scenarios(scenarios)
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
       
  

        
  
    
    # --- real instance with all scenarios ---
    f = FAIR()
    all_scenarios = [baseline_scenario] + list(replacement_dicts.keys())
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
    if not CH4_lifetime_effects:
        for specie in ["NOx","CO","VOC"]:
            f.species_configs["ch4_lifetime_chemical_sensitivity"]\
                .loc[dict(specie=specie)] =0.
    for s in all_scenarios:
        f.emissions.loc[dict(scenario=s)] = baseline_emissions
        f.forcing.loc[dict(scenario=s)] = baseline_forcing
    
    for s, replacement in replacement_dicts.items():
        for specie, replacement_scenario in replacement.items():
            f.emissions.loc[dict(scenario=s, specie=specie)] = f_tmp.emissions.loc[dict(scenario=replacement_scenario, specie=specie)].copy()
    
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)
    setattr(f,"baseline_scenario",baseline_scenario)
    if run_model:
        f.run()
    return f
    
def get_CMIP7_emissions(scenario,species):
    emissions_file=_DATA_DIR / 'emissions/extensions_1750-2500.csv'
    edf=pd.read_csv(emissions_file)
    
    
    if type(species) == type("this"):
        species=[species]
    condition = np.logical_and(edf.scenario==scenario , edf.variable.isin(species))
    subdf=edf[condition]
   
    units=subdf.iloc[:,4].values[0]
    fairspecies=subdf.variable.values
    years=[float(x) for x in subdf.iloc[:,5:].columns]
    
    da=xr.DataArray(subdf.iloc[:,5:].values,coords=dict(specie=fairspecies,timepoints=years),attrs=dict(units=units))
    del edf
    return da

def _normalize_solution_name(name):
    """
    Squash a solution name down for fuzzy matching across the different
    spreadsheets this module reads: lowercase, expand '&' to 'and' (the
    Explorer website and reduced_emissions.csv don't agree on which to use,
    e.g. "Paper & Cardboard" vs "Paper and Cardboard", "Grasslands &
    Savannas" vs "Grasslands and Savannas"), then strip punctuation.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower().replace("&", " and "))

def achievable_adoption(solution,bound):
    """
    Get low or high range of achievable adoption (scraped from Explorer website)

    Parameters:
        solution: doesn't need to match the website's spelling exactly --
            matched after normalizing both sides (see _normalize_solution_name),
            with a prefix-match fallback for names like "Improve Nonmotorized
            Transportation TOTAL Nonmotorized travel" that only partially
            overlap the website's "Improve Nonmotorized Transportation".
        bound: one of "low" or "high" (case-insensitive)
    """
    if bound.lower() == "low":
        col="Low Achievable Range"
    else:
        col="High Achievable Range"

    adoption_df=pd.read_excel(_DATA_DIR / "explorer-website/drawdown_adoption_ranges.xlsx")
    key=_normalize_solution_name(solution)
    normalized=adoption_df["Solution"].map(_normalize_solution_name)

    exact=adoption_df[normalized == key]
    if len(exact) == 1:
        return float(exact[col].values[0])

    prefix=adoption_df[[key.startswith(n) or n.startswith(key) for n in normalized]]
    if len(prefix) == 1:
        return float(prefix[col].values[0])
    if len(prefix) > 1:
        raise ValueError(f"'{solution}' matches multiple rows in drawdown_adoption_ranges.xlsx: "
                         f"{list(prefix['Solution'])}")
    raise ValueError(f"'{solution}' not found in drawdown_adoption_ranges.xlsx "
                     "(checked exact and prefix match after normalizing '&'/'and' and punctuation)")

def scenarios_to_emissions(baseline_scenario,scenariodict):
    """
    Translate between CMIP7 scenarios and emissions for Drawdown solution model
    """
    emissions_file=replacement_model(baseline_scenario,scenariodict,lite=True,n_lite=1,run_model=False)

    perturbation_dicts={}
    baseline_emissions=emissions_file.emissions.sel(scenario=baseline_scenario)
    delta_emissions = emissions_file.emissions-baseline_emissions
    for scenario in delta_emissions.scenario.values:
        if scenario != baseline_scenario:
            perturbation_dicts[scenario]={}
            scen_em=delta_emissions.sel(scenario=scenario).isel(config=0).dropna(dim="specie")
            pert_species=scen_em.specie[scen_em.sum(dim="timepoints")!=0]
            for specie in pert_species.values:
                perturbation_dicts[scenario][specie]=scen_em.sel(specie=specie).values
    return perturbation_dicts
           