"""
Simple stacked landfill-methane solutions for drawdown_model.

Four solutions act on one shared pool of landfill CH4, E0 (Mt CH4/yr):

    C(t)        = methane kept out of landfills by composting AND by food
                  loss & waste (FLW) that is never produced, spread over
                  time with first-order decay (single global k). Both
                  streams use one CH4 factor per DRY tonne; FLW tonnes are
                  converted from wet to dry with the moisture content.
    fc(t)       = fraction of remaining landfill CH4 captured
    fb(t)       = fraction of what is left oxidised by biocovers

    emitted(t)  = (E0 - C(t)) * (1 - fc(t)) * (1 - fb(t))
    reduction   = E0 - emitted

The reductions multiply rather than add, so stacking never removes more
methane than exists in the pool.

Reducing FLW also avoids the emissions of producing that food (farm, supply
chain). Drawdown reports these only as an aggregate 2.82 t CO2e(100-yr) per
tonne, with no gas split and no landfill component, so they are split into
CH4, N2O and CO2 using assumed shares and added on top, with no lag.
"""
import numpy as np

TIMEPOINTS = np.arange(1750, 2102)   # matches get_input / drawdown_model

# AR6 100-yr GWPs, used only to split the FLW production emissions
GWP100 = {"CH4": 27.0, "N2O": 273.0}


def waste_solutions(composting=None,
                    capture=None,
                    biocovers=None,
                    food_loss_waste=None,
                    E0=69.56,
                    ch4_per_dry_tonne=0.10,
                    moisture=0.6,
                    k=0.15,
                    flw_landfill_fraction=0.3,
                    flw_co2e100_per_tonne=2.82,
                    flw_gas_shares=(0.35, 0.10, 0.55),
                    flw_co2_specie="CO2 FFI",
                    scenario_name="waste"):
    """
    Build a perturbation dict for explorer.drawdown_model.

    Parameters
    ----------
    composting : array over TIMEPOINTS, or None
        Additional DRY tonnes of organic waste diverted from landfill each year.
    capture : array over TIMEPOINTS (values 0-1), or None
        Fraction of remaining landfill CH4 captured. To convert from
        Drawdown's "Mt CH4 abated", divide by E0.
    biocovers : array over TIMEPOINTS (values 0-1), or None
        Fraction of uncaptured CH4 oxidised by biocovers (same conversion).
    food_loss_waste : array over TIMEPOINTS, or None
        Tonnes of food loss and waste avoided each year (Drawdown unit,
        assumed WET; high achievable = 875e6 t/yr, a 50% cut).
    E0 : float
        Baseline landfill CH4 emissions, Mt/yr (constant-2024 baseline).
    ch4_per_dry_tonne : float
        Lifetime CH4 EMITTED per DRY tonne of organic waste landfilled (t/t),
        used for both composting and FLW. 0.10 matches the Drawdown
        composting row (an emitted, global-average value).
    moisture : float
        Moisture content used to convert FLW wet tonnes to dry (Drawdown
        assumes 0.6).
    k : float
        Single global first-order decay rate (1/yr), shared by both streams.
    flw_landfill_fraction : float
        Share of avoided FLW that would otherwise have been landfilled or
        dumped. Losses on farms and in processing mostly are not, so this is
        well below 1. ASSUMPTION, roughly 0.2-0.4.
    flw_co2e100_per_tonne : float
        Production emissions avoided per tonne of FLW, t CO2e(100-yr).
        Drawdown median 2.82 (IQR 2.75-3.30); excludes waste management.
    flw_gas_shares : (CH4, N2O, CO2) shares of flw_co2e100_per_tonne
        on a 100-yr basis. Drawdown gives no split, so this is an
        ASSUMPTION; vary it as a sensitivity.
    flw_co2_specie : "CO2 FFI" or "CO2 AFOLU"
    scenario_name : str

    Returns
    -------
    {scenario_name: {specie: array}}. CH4 and N2O in Mt/yr, CO2 in Gt/yr;
    negative = avoided.

    Notes
    -----
    Composting and FLW draw on the same organic-waste stream. At high
    adoption of both, check that the combined tonnes do not exceed the
    organic waste actually landfilled; the cap at E0 only limits methane.
    FLW production emissions overlap with Improve Diets if the avoided food
    is ruminant meat; avoid counting both at high adoption.
    """
    n = len(TIMEPOINTS)
    zeros = np.zeros(n)

    # lifetime landfill CH4 kept out each year (Mt), from both streams
    lifetime_ch4 = zeros.copy()
    if composting is not None:
        lifetime_ch4 += np.asarray(composting, float) * ch4_per_dry_tonne * 1e-6
    if food_loss_waste is not None:
        flw = np.asarray(food_loss_waste, float)                 # wet t/yr
        flw_dry_landfilled = flw * (1 - moisture) * flw_landfill_fraction
        lifetime_ch4 += flw_dry_landfilled * ch4_per_dry_tonne * 1e-6

    # spread over subsequent years by first-order decay (kernel sums to ~1)
    years = np.arange(n)
    kernel = np.exp(-k * years) - np.exp(-k * (years + 1))
    C = np.minimum(np.convolve(lifetime_ch4, kernel)[:n], E0)

    fc = zeros if capture is None else np.clip(capture, 0, 1)
    fb = zeros if biocovers is None else np.clip(biocovers, 0, 1)

    emitted = (E0 - C) * (1 - fc) * (1 - fb)
    pert = {"CH4": -(E0 - emitted)}

    # FLW production emissions avoided immediately
    if food_loss_waste is not None:
        s_ch4, s_n2o, s_co2 = flw_gas_shares
        co2e = flw * flw_co2e100_per_tonne                      # t CO2e100/yr
        pert["CH4"] = pert["CH4"] - co2e * s_ch4 / GWP100["CH4"] * 1e-6   # Mt
        pert["N2O"] = -co2e * s_n2o / GWP100["N2O"] * 1e-6              # Mt
        pert[flw_co2_specie] = -co2e * s_co2 * 1e-9                     # Gt

    return {scenario_name: pert}