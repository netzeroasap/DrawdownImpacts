"""
Time-varying avoided-emissions calculations for Project Drawdown's "Protect X"
land/ocean ecosystem solutions (forests, peatlands, seaweed ecosystems,
grasslands and savannas).

Background
----------
Each "Protect X" solution compares two futures for a given hectare: a
BASELINE future where the ecosystem keeps degrading at its historical rate,
and a PROTECTED future where degradation is slower (but usually not zero --
protection isn't 100% effective). Degrading a hectare typically:
  1. releases a one-time carbon stock (vegetation/biomass, sometimes soil
     carbon too), and
  2. turns an ongoing carbon SINK (sequestration) into, for some ecosystems,
     an ongoing carbon/CH4/N2O SOURCE.

The explorer's built-in treatment collapses all of this into a single flat
number: (stock + sequestration*30) * (h_baseline - h_protected), applied
every year forever. That is an overestimate -- see the markdown cells in
notebooks/LandSolutions.ipynb for the derivation -- because it ignores that
both the baseline and protected hectares are actually being depleted over
time, not held at their initial condition. `ecosystem_emissions` below does
the full time-varying integral instead.

Not every ecosystem's spreadsheet gives us the same raw ingredients, though
(see each read_*_data docstring for specifics), so this module doesn't force
a single interface across ecosystems where the data doesn't support it:
  - forests: raw cumulative-hectares-lost + area data available -> supports
    re-deriving h_baseline/h_protected under either assumption below.
  - peatlands, seaweed: only an already-computed rate is available -> the
    deforestation_model choice only affects how that same rate propagates
    through time, not how it's derived.
  - grasslands: no baseline hazard rate at all, only an avoided-loss rate,
    and the spreadsheet's own "effectiveness" table already bakes in the
    flat/naive calculation this module elsewhere replaces. grassland_emissions
    is therefore NOT built on the shared ecosystem_emissions engine -- see its
    docstring.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "data" / "zenodo_spreadsheets"

FAIR_START_TIME = 1750
FAIR_END_TIME = 2101


# ---------------------------------------------------------------------------
# Shared time-integration engine
# ---------------------------------------------------------------------------

def _survival_and_clearing_rate(h, tsteps, deforestation_model):
    """
    h: annual degradation rate (fraction/year); meaning depends on
       deforestation_model (see ecosystem_emissions).
    tsteps: years elapsed since implementation_start (clipped to >=0).
    Returns (survival, clearing_rate) arrays, same shape as tsteps.
    """
    if deforestation_model == "constant_rate":
        # exponential decline: a constant FRACTION of the remaining area is
        # degraded every year
        survival = np.exp(-h * tsteps)
        clearing_rate = h * survival
    elif deforestation_model == "constant_acreage":
        # linear decline: a constant fraction of the ORIGINAL area is
        # degraded every year, until none is left
        survival = np.clip(1.0 - h * tsteps, 0.0, 1.0)
        clearing_rate = np.where(survival > 0, h, 0.0)
    else:
        raise ValueError(
            f"Unknown deforestation_model {deforestation_model!r}; "
            "expected 'constant_rate' or 'constant_acreage'"
        )
    return survival, clearing_rate


def ecosystem_emissions(h_baseline, h_protected, stock_ef, ongoing_efs,
                         implementation_start, deforestation_model="constant_rate"):
    """
    Shared time-integration engine behind forest_emissions/peatland_emissions/
    seaweed_emissions. Computes avoided emissions per hectare per year from
    protecting one ecosystem type, given:

    h_baseline, h_protected: annual baseline/protected degradation rate
        (fraction of area lost per year). h_protected < h_baseline (protection
        isn't 100% effective, just slower degradation).
    stock_ef: one-time carbon stock (tCO2e/ha) released at the moment a
        hectare is degraded/cleared/drained.
    ongoing_efs: dict {name: EF value} of per-year effect sizes (tCO2e/ha/yr
        for CO2 terms, or raw gas mass/ha/yr for CH4/N2O terms) that apply to
        every hectare that stays intact under protection relative to
        baseline -- e.g. lost sequestration, or avoided ongoing emissions
        from land that would otherwise already be degraded. Both kinds of
        term end up proportional to the same "intact-area gap" between
        scenarios (survival_protected - survival_baseline), so they're
        combined the same way regardless of their physical origin -- see
        the LandSolutions.ipynb markdown derivation.
    deforestation_model: "constant_rate" (default, exponential decline -- h
        acts as a constant proportional hazard rate) or "constant_acreage"
        (linear decline -- h acts as a constant fraction of the ORIGINAL
        area lost per year, until none is left).

    Returns a dict with per-year arrays (length FAIR_END_TIME-FAIR_START_TIME+1,
    zeroed out before implementation_start):
        stock_term:              one-time-stock-loss term, already differenced
                                  (baseline minus protected)
        ongoing_terms:           dict {name: array}, one per ongoing_efs key,
                                  already differenced
        survival_baseline/protected, clearing_rate_baseline/protected:
                                  the raw building blocks, exposed for
                                  debugging/plotting
        area_gap:                survival_protected - survival_baseline
    """
    timepoints = np.arange(FAIR_START_TIME, FAIR_END_TIME + 1, 1)
    t0 = implementation_start
    tsteps = np.clip(timepoints - t0, a_min=0, a_max=None).astype(float)
    in_window = (timepoints >= t0)

    survival_baseline, clearing_rate_baseline = _survival_and_clearing_rate(
        h_baseline, tsteps, deforestation_model)
    survival_protected, clearing_rate_protected = _survival_and_clearing_rate(
        h_protected, tsteps, deforestation_model)

    area_gap = survival_protected - survival_baseline
    stock_term = np.where(
        in_window, stock_ef * (clearing_rate_baseline - clearing_rate_protected), 0.0)
    ongoing_terms = {
        name: np.where(in_window, ef * area_gap, 0.0)
        for name, ef in ongoing_efs.items()
    }

    return dict(
        stock_term=stock_term,
        ongoing_terms=ongoing_terms,
        survival_baseline=survival_baseline,
        survival_protected=survival_protected,
        clearing_rate_baseline=clearing_rate_baseline,
        clearing_rate_protected=clearing_rate_protected,
        area_gap=area_gap,
    )


# ---------------------------------------------------------------------------
# Forests
# ---------------------------------------------------------------------------

def read_forest_data(deforestation_model="constant_rate", loss_period_years=None):
    """
    Reads carbon stock/sequestration and forest-loss hazard rates by climate
    zone from the Protect Forests spreadsheet.

    deforestation_model:
        "constant_rate"     (default) - h_baseline/h_protected taken directly
                             from the spreadsheet, which reports them as a
                             constant proportional hazard rate.
        "constant_acreage"  - re-derives h_baseline/h_protected assuming a
                             constant NUMBER OF HECTARES (not a constant
                             fraction) is cleared each year instead:
                                 annual_loss_ha_baseline
                                     = forest_loss_ha_total_2001_2022 / loss_period_years
                                 annual_loss_ha_protected
                                     = annual_loss_ha_baseline * (1 - relative_reduction_from_protection)
                             expressed here as the equivalent fraction of
                             forest_area_outside_PA_ha lost per year, so the
                             returned dataframe stays drop-in compatible.
                             The raw hectare figures are also kept, in
                             annual_loss_ha_baseline/annual_loss_ha_protected.

    loss_period_years: number of years spanned by the spreadsheet's
                        cumulative "total forest cover loss" figure (only
                        used for deforestation_model="constant_acreage"). If
                        None (default), parsed automatically from the
                        column's own header, e.g. "Total forest cover loss
                        outside of PAs (2001-2022)" -> 22 years.
    """
    path = _DATA_DIR / "Protect Forests- Solution Assessment Spreadsheet.xlsx"
    sheet = "Detailed effectiveness data"

    # --- Table 1: aggregate carbon stock / sequestration by climate zone ---
    carbon = pd.read_excel(path, sheet_name=sheet, skiprows=17, nrows=4, usecols="A:E")
    carbon.columns = [
        "climate_zone", "C_stock_tCO2e_ha",
        "C_seq_30yr_tCO2e_ha", "C_seq_aboveground_30yr", "C_seq_belowground_30yr",
    ]
    carbon["climate_zone"] = carbon["climate_zone"].str.strip().str.lower()
    carbon["C_seq_annual_tCO2e_ha_yr"] = carbon["C_seq_30yr_tCO2e_ha"] / 30.0

    # --- Table 2: aggregate forest loss / hazard rates by climate zone ---
    hazard_header = pd.read_excel(
        path, sheet_name=sheet, skiprows=57, nrows=1, usecols="A:H", header=None)
    hazard = pd.read_excel(path, sheet_name=sheet, skiprows=57, nrows=4, usecols="A:H")
    hazard.columns = [
        "cause", "climate_zone", "forest_loss_ha_total_2001_2022",
        "forest_area_outside_PA_ha", "relative_reduction_from_protection",
        "h_baseline", "h_protected", "delta_h",
    ]
    hazard["climate_zone"] = hazard["climate_zone"].str.strip().str.lower()
    hazard["climate_zone"] = hazard["climate_zone"].replace({
        "subtropic": "subtropical", "tropic": "tropical",
    })

    if deforestation_model == "constant_rate":
        pass
    elif deforestation_model == "constant_acreage":
        if loss_period_years is None:
            header_text = str(hazard_header.iloc[0, 2])
            years = [int(y) for y in re.findall(r"(\d{4})", header_text)]
            if len(years) != 2:
                raise ValueError(
                    f"Couldn't infer loss_period_years from header {header_text!r}; "
                    "pass loss_period_years explicitly."
                )
            loss_period_years = years[1] - years[0] + 1

        annual_loss_ha_baseline = hazard["forest_loss_ha_total_2001_2022"] / loss_period_years
        annual_loss_ha_protected = annual_loss_ha_baseline * (1 - hazard["relative_reduction_from_protection"])
        hazard["annual_loss_ha_baseline"] = annual_loss_ha_baseline
        hazard["annual_loss_ha_protected"] = annual_loss_ha_protected
        hazard["h_baseline"] = annual_loss_ha_baseline / hazard["forest_area_outside_PA_ha"]
        hazard["h_protected"] = annual_loss_ha_protected / hazard["forest_area_outside_PA_ha"]
        hazard["delta_h"] = hazard["h_baseline"] - hazard["h_protected"]
    else:
        raise ValueError(
            f"Unknown deforestation_model {deforestation_model!r}; "
            "expected 'constant_rate' or 'constant_acreage'"
        )

    df = pd.merge(hazard, carbon, on="climate_zone", how="inner")
    df["avoided_emissions_check"] = df["delta_h"] * df["C_stock_tCO2e_ha"]
    df["sequestration_check"] = df["delta_h"] * df["C_seq_30yr_tCO2e_ha"]
    df["total_effectiveness_check"] = df["avoided_emissions_check"] + df["sequestration_check"]
    return df


def forest_emissions(climate_zone, implementation_start, deforestation_model="constant_rate", debug=False):
    """
    Avoided CO2 emissions (tCO2e/ha/yr) from protecting a forest, as a
    function of years since implementation_start. See ecosystem_emissions
    for the deforestation_model options.
    """
    df = read_forest_data(deforestation_model=deforestation_model)
    df_b = df[df["climate_zone"] == climate_zone]
    Cstore = df_b["C_stock_tCO2e_ha"].values[0]
    Cseq   = df_b["C_seq_annual_tCO2e_ha_yr"].values[0]
    hb = df_b["h_baseline"].values[0]
    hp = df_b["h_protected"].values[0]

    eng = ecosystem_emissions(hb, hp, Cstore, {"seq": Cseq}, implementation_start, deforestation_model)
    if debug:
        return dict(
            Cstore_base=Cstore * eng["clearing_rate_baseline"],
            Cstore_prot=Cstore * eng["clearing_rate_protected"],
            Cseq_base=Cseq * eng["survival_baseline"],
            Cseq_prot=Cseq * eng["survival_protected"],
        )
    return eng["stock_term"] + eng["ongoing_terms"]["seq"]


# ---------------------------------------------------------------------------
# Peatlands
# ---------------------------------------------------------------------------

def read_peatland_data():
    """
    Reads emissions factors and hazard rates by climate zone from the
    Protect Peatlands spreadsheet.

    Unlike read_forest_data, this has no deforestation_model kwarg: the
    peatlands hazard table only reports "Mean annual peatland loss" as an
    already-computed rate (h_baseline/h_protected), with no raw
    cumulative-loss/area pair to re-derive it from. The constant_rate vs
    constant_acreage choice for peatlands lives entirely in
    peatland_emissions, which reinterprets this same h_baseline/h_protected
    number under either functional form.
    """
    path = _DATA_DIR / "Protect Peatlands- Solution Assessment Spreadsheet.xlsx"
    sheet = "Detailed emissions factor data"
    gwp_sheet = "conversions global warming pote"

    gwp = pd.read_excel(path, sheet_name=gwp_sheet, skiprows=1, nrows=2, usecols="A:C")
    gwp.columns = ["gas", "GWP100", "GWP20"]
    GWP100_CH4 = gwp.loc[gwp.gas == "CH4", "GWP100"].values[0]
    GWP100_N2O = gwp.loc[gwp.gas == "N2O", "GWP100"].values[0]

    ef = pd.read_excel(path, sheet_name=sheet, skiprows=55, nrows=4, usecols="A:K")
    ef.columns = [
        "climate_zone",
        "C_stock_tCO2e_ha",              # EF, CO2 from initial vegetation (one-time)
        "C_peat_oxidation_tCO2e_ha_yr",  # EF, CO2 from peat (per year)
        "C_DOC_tCO2e_ha_yr",             # EF, dissolved organic carbon transport (per year)
        "M_onfield_20yr_tCO2e_ha_yr",
        "M_onfield_100yr_tCO2e_ha_yr",
        "frac_ditch_area",
        "M_ditch_20yr_tCO2e_ha_yr",
        "M_ditch_100yr_tCO2e_ha_yr",
        "N_flux_tCO2e_ha_yr",
        "C_seq_tCO2e_ha_yr",
    ]
    ef["climate_zone"] = ef["climate_zone"].str.strip().str.lower()

    ef["C_flux_tCO2e_ha_yr"] = ef["C_peat_oxidation_tCO2e_ha_yr"] + ef["C_DOC_tCO2e_ha_yr"]
    M_flux_100yr_tCO2e_ha_yr = ef["M_onfield_100yr_tCO2e_ha_yr"] + ef["M_ditch_100yr_tCO2e_ha_yr"]
    ef["CH4_flux_Mt_ha_yr"] = (M_flux_100yr_tCO2e_ha_yr / GWP100_CH4) / 1e6
    ef["N2O_flux_Mt_ha_yr"] = (ef["N_flux_tCO2e_ha_yr"] / GWP100_N2O) / 1e6
    ef = ef.drop(columns=[
        "M_onfield_20yr_tCO2e_ha_yr", "M_onfield_100yr_tCO2e_ha_yr",
        "M_ditch_20yr_tCO2e_ha_yr", "M_ditch_100yr_tCO2e_ha_yr",
        "N_flux_tCO2e_ha_yr",
    ])

    hazard = pd.read_excel(path, sheet_name=sheet, skiprows=86, nrows=4, usecols="A:E")
    hazard.columns = [
        "climate_zone", "h_baseline", "relative_reduction_from_protection",
        "h_protected", "delta_h",
    ]
    hazard["climate_zone"] = hazard["climate_zone"].str.strip().str.lower()

    for d in (ef, hazard):
        d["climate_zone"] = d["climate_zone"].replace({
            "subtropic": "subtropical", "tropic": "tropical",
        })

    return pd.merge(hazard, ef, on="climate_zone", how="inner")


def peatland_emissions(climate_zone, implementation_start, deforestation_model="constant_rate", debug=False):
    """
    Avoided emissions from protecting a peatland. Peatland protection avoids
    THREE gases, not one: draining/degrading a peatland (a) releases its
    vegetation carbon stock once, then (b) turns it into an ongoing source of
    CO2 (peat oxidation + dissolved organic carbon) and N2O, while (c) also
    removing the ongoing CO2 sequestration an intact peatland would have
    provided, and the CH4 that drainage ditches keep emitting even in an
    otherwise-drained field. So unlike forest_emissions -- which returns one
    tCO2e/ha/yr array -- this returns a dict of separate per-hectare-per-year
    trajectories, one per gas:
        {"CO2": tCO2e/ha/yr, "CH4": Mt CH4/ha/yr, "N2O": Mt N2O/ha/yr}
    ready to be scaled by adoption area and fed into per-species emissions
    inputs (e.g. FAIR).

    See ecosystem_emissions for the deforestation_model options; note
    read_peatland_data does not itself take this kwarg (see its docstring).
    """
    df = read_peatland_data()
    df_b = df[df["climate_zone"] == climate_zone]
    Cstock  = df_b["C_stock_tCO2e_ha"].values[0]
    Cflux   = df_b["C_flux_tCO2e_ha_yr"].values[0]
    Cseq    = df_b["C_seq_tCO2e_ha_yr"].values[0]
    CH4flux = df_b["CH4_flux_Mt_ha_yr"].values[0]
    N2Oflux = df_b["N2O_flux_Mt_ha_yr"].values[0]
    hb = df_b["h_baseline"].values[0]
    hp = df_b["h_protected"].values[0]

    eng = ecosystem_emissions(
        hb, hp, Cstock, {"seq": Cseq, "flux": Cflux, "CH4": CH4flux, "N2O": N2Oflux},
        implementation_start, deforestation_model)
    ot = eng["ongoing_terms"]

    if debug:
        return dict(
            stock_term=eng["stock_term"], seq_term=ot["seq"], flux_term=ot["flux"],
            CH4_term=ot["CH4"], N2O_term=ot["N2O"],
            survival_baseline=eng["survival_baseline"], survival_protected=eng["survival_protected"],
        )
    return dict(
        CO2=eng["stock_term"] + ot["seq"] + ot["flux"],
        CH4=ot["CH4"],
        N2O=ot["N2O"],
    )


# ---------------------------------------------------------------------------
# Seaweed ecosystems
# ---------------------------------------------------------------------------

def read_seaweed_data():
    """
    Reads biomass carbon stock/sequestration by macroalgae type, and the
    (single, ecosystem-wide) loss/protection rate, from the Protect Seaweed
    Ecosystems spreadsheet.

    Unlike forests/peatlands/grasslands, seaweed ecosystems aren't broken out
    by climate zone -- there's only one global loss rate ("due to limited
    data, we assume [kelp's] loss rate... applies to all macroalgae groups
    analyzed"), applied uniformly to each macroalgae_type row here. Like
    peatlands, there's no raw hectares/area pair to re-derive it from, so
    (as with read_peatland_data) there's no deforestation_model kwarg here --
    that choice lives in seaweed_emissions.
    """
    path = _DATA_DIR / "Protect Seaweed Ecosystems- Solution Assessment Spreadsheet.xlsx"

    loss = pd.read_excel(path, sheet_name="Detailed loss data", skiprows=2, nrows=1, usecols="A:D")
    loss.columns = ["seaweed_type", "h_baseline", "protection_effectiveness", "delta_h"]
    h_baseline = loss["h_baseline"].values[0]
    delta_h = loss["delta_h"].values[0]
    h_protected = h_baseline - delta_h

    ef = pd.read_excel(path, sheet_name="Detailed effectiveness data", skiprows=25, nrows=2, usecols="A:G")
    ef.columns = [
        "macroalgae_type",
        "C_stock_tCO2e_ha", "C_stock_25th", "C_stock_75th",
        "C_seq_30yr_tCO2e_ha", "C_seq_30yr_25th", "C_seq_30yr_75th",
    ]
    ef["macroalgae_type"] = ef["macroalgae_type"].str.strip().str.lower()
    ef["C_seq_annual_tCO2e_ha_yr"] = ef["C_seq_30yr_tCO2e_ha"] / 30.0

    ef["h_baseline"] = h_baseline
    ef["h_protected"] = h_protected
    ef["delta_h"] = delta_h
    return ef


def seaweed_emissions(macroalgae_type, implementation_start, deforestation_model="constant_rate", debug=False):
    """
    Avoided CO2 emissions (tCO2e/ha/yr) from protecting a seaweed ecosystem
    (kelp/macroalgae), by macroalgae_type (e.g. "subtidal brown",
    "subtidal deep red (excl. rhodoliths)" -- see read_seaweed_data()).
    Only one gas is tracked for this ecosystem (biomass carbon), so -- like
    forest_emissions -- this returns a single array, not a per-gas dict.
    See ecosystem_emissions for the deforestation_model options.
    """
    df = read_seaweed_data()
    df_b = df[df["macroalgae_type"] == macroalgae_type]
    Cstock = df_b["C_stock_tCO2e_ha"].values[0]
    Cseq   = df_b["C_seq_annual_tCO2e_ha_yr"].values[0]
    hb = df_b["h_baseline"].values[0]
    hp = df_b["h_protected"].values[0]

    eng = ecosystem_emissions(hb, hp, Cstock, {"seq": Cseq}, implementation_start, deforestation_model)
    if debug:
        return dict(
            Cstock_base=Cstock * eng["clearing_rate_baseline"],
            Cstock_prot=Cstock * eng["clearing_rate_protected"],
            Cseq_base=Cseq * eng["survival_baseline"],
            Cseq_prot=Cseq * eng["survival_protected"],
        )
    return eng["stock_term"] + eng["ongoing_terms"]["seq"]


# ---------------------------------------------------------------------------
# Grasslands and savannas
# ---------------------------------------------------------------------------

def read_grassland_data():
    """
    Reads the final "Effectiveness by climate zone" table from the Protect
    Grasslands and Savannas spreadsheet.

    This ecosystem does NOT fit the hazard-rate framework used above: the
    spreadsheet's "Grassland loss rates" table gives only an avoided-loss
    rate (delta_h) -- there's no h_baseline/h_protected split at all, so
    ecosystem_emissions can't be used (it needs both). The spreadsheet's own
    "Effectiveness by climate zone" table already bakes in the flat
    (avoided_rate * emissions_factor) calculation this module elsewhere
    replaces with a time-varying integral -- see grassland_emissions.

    N2O is converted from the spreadsheet's tCO2e figure back to raw mass
    (Mt N2O/ha/yr) via GWP100, for consistency with peatland_emissions'
    units convention.
    """
    path = _DATA_DIR / "Protect Grasslands and Savannas- Solution Assessment Spreadsheet.xlsx"
    sheet = "Detailed effectiveness data"

    gwp = pd.read_excel(path, sheet_name="conversions global warming pote", skiprows=1, nrows=2, usecols="A:C")
    gwp.columns = ["gas", "GWP100", "GWP20"]
    GWP100_N2O = gwp.loc[gwp.gas == "N2O", "GWP100"].values[0]

    eff = pd.read_excel(path, sheet_name=sheet, skiprows=86, nrows=4, usecols="A:J")
    eff.columns = [
        "climate_zone",
        "CO2_all_sources", "CO2_all_sources_25th", "CO2_all_sources_75th",
        "lost_C_seq", "lost_C_seq_25th", "lost_C_seq_75th",
        "N2O_tCO2e", "N2O_tCO2e_25th", "N2O_tCO2e_75th",
    ]
    eff["climate_zone"] = eff["climate_zone"].str.strip().str.lower()
    eff["climate_zone"] = eff["climate_zone"].replace({
        "subtropic": "subtropical", "tropic": "tropical",
    })
    eff["CO2_effectiveness_tCO2e_ha_yr"] = eff["CO2_all_sources"] + eff["lost_C_seq"]
    eff["N2O_effectiveness_Mt_ha_yr"] = (eff["N2O_tCO2e"] / GWP100_N2O) / 1e6
    return eff


def grassland_emissions(climate_zone, implementation_start, debug=False):
    """
    Avoided emissions from protecting grassland/savanna, by climate zone.

    Unlike forest_emissions/peatland_emissions/seaweed_emissions, this takes
    no deforestation_model kwarg and isn't built on ecosystem_emissions --
    the source data doesn't support a time-varying hazard-rate calculation
    (see read_grassland_data). Instead this is a step function: 0 before
    implementation_start, then the spreadsheet's own flat per-hectare-per-year
    effectiveness value every year after, for as long as protection holds.

    Returns a dict (grasslands track two gases, like peatlands):
        {"CO2": tCO2e/ha/yr, "N2O": Mt N2O/ha/yr}
    """
    df = read_grassland_data()
    df_b = df[df["climate_zone"] == climate_zone]
    CO2_per_ha_yr = df_b["CO2_effectiveness_tCO2e_ha_yr"].values[0]
    N2O_per_ha_yr = df_b["N2O_effectiveness_Mt_ha_yr"].values[0]

    timepoints = np.arange(FAIR_START_TIME, FAIR_END_TIME + 1, 1)
    in_window = (timepoints >= implementation_start)
    avoided_CO2 = np.where(in_window, CO2_per_ha_yr, 0.0)
    avoided_N2O = np.where(in_window, N2O_per_ha_yr, 0.0)

    if debug:
        return dict(CO2_per_ha_yr=CO2_per_ha_yr, N2O_per_ha_yr=N2O_per_ha_yr)
    return dict(CO2=avoided_CO2, N2O=avoided_N2O)


# ---------------------------------------------------------------------------
# FAIR input builder
# ---------------------------------------------------------------------------

# Per-gas conversion from this module's native units to what FAIR wants
# (checked against fair.structure.units.desired_emissions_units). CO2 here is
# always tCO2e/ha/yr (tons), and FAIR wants Gt CO2/yr, hence 1e-9. CH4/N2O are
# already raw gas mass in Mt/ha/yr (read_peatland_data/read_grassland_data did
# the GWP-based conversion back to mass when the spreadsheet only gave a CO2e
# figure) -- exactly what FAIR's Mt CH4/yr and Mt N2O/yr channels want, so no
# further scaling.
_LAND_UNIT_SCALE = {"CO2": 1e-9, "CH4": 1.0, "N2O": 1.0}
_LAND_FAIR_SPECIE_NAME = {"CO2": "CO2 AFOLU", "CH4": "CH4", "N2O": "N2O"}


def get_land_input(ecosystem_fn, group_value, adoption_curve, max_hectares,
                    implementation_start, function_kwargs={}, ecosystem_kwargs={},
                    scenario_name=None):
    """
    Builds a FAIR-ready perturbation dict for a "Protect X" land/ocean
    solution -- analogous to drawdown.explorer.get_input, but adapted for the
    fact that protecting a hectare here has a genuinely TIME-VARYING benefit.

    get_input assumes a constant per-unit annual benefit once something is
    adopted, so it can just take the adoption curve's output and scale it
    directly. That assumption doesn't hold here: forest_emissions/
    peatland_emissions/etc. all show avoided emissions per hectare depend on
    how long ago THAT hectare was protected (rising for decades as the
    baseline/protected survival curves diverge -- see the LandSolutions.ipynb
    discussion), not on calendar time. So instead of scaling adoption_curve's
    output directly, this convolves the FLOW of newly-protected hectares each
    year against each hectare-cohort's own avoided-emissions trajectory:

        total(T) = sum over s<=T of new_ha(s) * Et(T - s)

    where new_ha(s) is hectares newly protected in year s, and Et(tau) is the
    per-hectare avoided-emissions rate tau years after protection began (the
    "kernel", built by calling ecosystem_fn with implementation_start pinned
    to FAIR_START_TIME so its own elapsed-time axis lines up index-for-index
    with tau). As a sanity check: with adoption_curve=step_function (adopt
    everything at once), this reduces exactly to
    max_hectares * ecosystem_fn(group_value, implementation_start).

    Parameters
    ----------
    ecosystem_fn: one of forest_emissions, peatland_emissions,
        seaweed_emissions, grassland_emissions.
    group_value: climate_zone (forest/peatland/grassland) or macroalgae_type
        (seaweed) -- whatever ecosystem_fn's first argument expects.
    adoption_curve: a CUMULATIVE adoption-curve function -- hectares
        protected BY each year, as a monotonically non-decreasing "stock"
        curve -- called as
        adoption_curve(timepoints, implementation_start, max_hectares, **function_kwargs).
        Use drawdown.explorer.step_function / linear_ramp / S_curve.
        explorer.pulse/rectangular_pulse describe a release that later
        REVERTS (drops back to/toward zero), which isn't physically
        meaningful for land protection -- avoid them here, since
        differencing them would imply hectares getting un-protected.
    max_hectares: the ceiling/asymptotic hectares protected (adoption_curve's
        L argument).
    implementation_start: year the protection ramp-up begins.
    function_kwargs: extra kwargs forwarded to adoption_curve (e.g. k,
        years_to_half_adoption for S_curve; matches get_input's naming).
    ecosystem_kwargs: extra kwargs forwarded to ecosystem_fn (e.g.
        deforestation_model="constant_acreage"). grassland_emissions doesn't
        take deforestation_model -- leave this {} for grasslands.
    scenario_name: dict key in the returned perturbation dict; defaults to
        "{ecosystem_fn.__name__}: {group_value}".

    Returns
    -------
    {scenario_name: {specie: array}}, in exactly the shape
    drawdown.explorer.drawdown_model's perturbation_dicts expects: species
    already named "CO2 AFOLU"/"CH4"/"N2O", scaled to FAIR's native units, and
    signed as a REDUCTION (negative), ready to add onto a baseline scenario.
    """
    timepoints = np.arange(FAIR_START_TIME, FAIR_END_TIME + 1, 1)

    # cumulative hectares protected by each year -> flow of NEWLY protected
    # hectares each year (what actually gets convolved against each cohort's
    # own avoided-emissions trajectory)
    cumulative_ha = adoption_curve(timepoints, implementation_start, max_hectares, **function_kwargs)
    new_ha = np.diff(cumulative_ha, prepend=0.0)

    # per-hectare avoided-emissions kernel(s), one per gas ecosystem_fn
    # tracks. Pinning implementation_start=FAIR_START_TIME makes the
    # kernel's own internal tsteps equal elapsed years since protection,
    # index-for-index -- i.e. kernel[tau] = Et(tau).
    kernels = ecosystem_fn(group_value, FAIR_START_TIME, **ecosystem_kwargs)
    if not isinstance(kernels, dict):
        kernels = {"CO2": kernels}

    perturbation = {}
    for gas, kernel in kernels.items():
        total = np.convolve(new_ha, kernel, mode="full")[:len(timepoints)]
        specie = _LAND_FAIR_SPECIE_NAME.get(gas, gas)
        perturbation[specie] = -1.0 * _LAND_UNIT_SCALE.get(gas, 1.0) * total

    if scenario_name is None:
        scenario_name = f"{ecosystem_fn.__name__}: {group_value}"
    return {scenario_name: perturbation}
