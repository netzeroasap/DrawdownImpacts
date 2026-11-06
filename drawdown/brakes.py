"""
brakes: "emergency brake" perturbations -- fast-acting methane solutions
pulled together, grouped by how quickly they can be deployed.

Usage
-----
    from drawdown import brakes
    SPEED = brakes.make_speeds(fast=3, medium=7, slow=12)
    ADD = brakes.adoption_ceilings("high")
    perts = brakes.build_brakes(2027, speed=SPEED, add=ADD)
    land = brakes.build_land_brakes(2027, speed=SPEED, add=ADD)         # all land
    land = brakes.build_land_brakes(2027, ecosystem="forest", climate_zone="tropical")

Both return {scenario_name: {specie: array over TIMEPOINTS}}, ready for
explorer.drawdown_model. build_brakes has one entry per brake plus "Fast
brakes", "All waste" and "All brakes" subtotals; build_land_brakes has one
entry per ecosystem and climate zone plus "All forests"-style subtotals and
"All land protection".
"""
import numpy as np

from drawdown import explorer, kernels
from drawdown.explorer import S_curve, stack

t = kernels.TIMEPOINTS
n = len(t)

# --- logistic S-curve steepness per deployment speed class ---
SPEED_K = {"fast": 1.0,     # ~90% by start+5
           "medium": 0.5,   # ~90% by start+11
           "slow": 0.3}     # ~90% by start+19

E0 = 69.56          # Mt CH4/yr landfill pool (Drawdown / IEA 2025)
MOISTURE = 0.6      # wet -> dry conversion for composted organic waste
HERD_LAG_YR = 1.5   # diets: exponential lag tau

# ADD key -> name in drawdown_adoption_ranges.xlsx
SOLUTION_NAMES = {
    "Manage Oil & Gas Methane": "Manage Oil & Gas Methane",
    "Manage Coal Mine Methane": "Manage Coal Mine Methane",
    "landfill capture": "Improve Landfill Management: Methane capture",
    "landfill biocovers": "Improve Landfill Management: Biocovers",
    "composting": "Increase Centralized Composting",
    "food loss & waste": "Reduce Food Loss & Waste",
    "Improve Diets": "Improve Diets",
}

# --- land protection brakes ---
CLIMATE_ZONES = ["boreal", "subtropical", "temperate", "tropical"]

# ecosystem -> (kernels emissions function, Explorer website name, subtotal key)
LAND_ECOSYSTEMS = {
    "forest": (kernels.forest_emissions, "Protect Forests", "All forests"),
    "peatland": (kernels.peatland_emissions, "Protect Peatlands", "All peatlands"),
    "grassland": (kernels.grassland_emissions, "Protect Grasslands and Savannas", "All grasslands"),
    "seaweed": (kernels.seaweed_emissions, "Protect Seaweed Ecosystems", "All seaweed"),
}


def land_solution_names():
    """ADD keys for the land brakes: one per ecosystem and climate zone (seaweed is global)."""
    names = []
    for eco, (_, name, _) in LAND_ECOSYSTEMS.items():
        if eco == "seaweed":
            names.append(name)
        else:
            names += [f"{name}: {z.capitalize()}" for z in CLIMATE_ZONES]
    return names


def make_speeds(fast, medium, slow):
    """S_curve kwargs for each speed class, given years to half adoption."""
    years = {"fast": fast, "medium": medium, "slow": slow}
    return {s: {"years_to_half_adoption": years[s], "k": SPEED_K[s]} for s in SPEED_K}


def adoption_ceilings(bound="high"):
    """
    ADDITIONAL adoption at saturation for each brake, from the Explorer
    website's achievable adoption range, converted to what build_brakes and
    build_land_brakes expect. Current adoption is not subtracted.

    Parameters:
        bound: "low" or "high" achievable adoption
    """
    raw = {k: explorer.achievable_adoption(name, bound) for k, name in SOLUTION_NAMES.items()}
    # hectares protected, keyed by Explorer website name (e.g. "Protect Forests: Tropical")
    land = {name: explorer.achievable_adoption(name, bound) for name in land_solution_names()}
    return land | {
        # Mt CH4/yr abated. ADOPTION CEILINGS (~whole sector)
        "Manage Oil & Gas Methane": raw["Manage Oil & Gas Methane"],
        "Manage Coal Mine Methane": raw["Manage Coal Mine Methane"],
        # Mt CH4/yr abated -> fractions of the landfill pool
        "landfill capture": raw["landfill capture"] / E0,
        "landfill biocovers": raw["landfill biocovers"] / E0,
        # wet t organic waste/yr -> dry t/yr
        "composting": raw["composting"] * (1 - MOISTURE),
        # wet t/yr avoided
        "food loss & waste": raw["food loss & waste"],
        # kg ruminant meat avoided per yr
        "Improve Diets": raw["Improve Diets"],
    }


def curve(L, speed, start, speeds):
    return S_curve(t, start, L, **speeds[speed])


def build_brakes(start=2026, suffix="", speed=None, add=None):
    """
    One perturbation dict per brake, per speed class, and the full stack.

    Parameters:
        start: year the brakes are pulled
        suffix: appended to every scenario name
        speed: output of make_speeds; defaults to make_speeds(3, 7, 12)
        add: output of adoption_ceilings; defaults to adoption_ceilings("high")
    """
    if speed is None:
        speed = make_speeds(3, 7, 12)
    if add is None:
        add = adoption_ceilings("high")
    p = {}

    # FAST: delta kernel, straight from get_input
    for sol in ["Manage Oil & Gas Methane", "Manage Coal Mine Methane"]:
        p.update(explorer.get_input(sol, S_curve, add[sol], start,
                                    function_kwargs=speed["fast"],
                                    scenario_name=sol + suffix))

    # MEDIUM: landfill capture + biocovers (multiplicative on the pool)
    fc = curve(add["landfill capture"], "medium", start, speed)
    fb = curve(add["landfill biocovers"], "medium", start, speed)
    # SLOW: composting + FLW (first-order decay) -- same pool
    comp = curve(add["composting"], "slow", start, speed)
    flw = curve(add["food loss & waste"], "slow", start, speed)

    p.update(kernels.waste_solutions(capture=fc, E0=E0, scenario_name="Landfill capture" + suffix))
    p.update(kernels.waste_solutions(biocovers=fb, E0=E0, scenario_name="Landfill biocovers" + suffix))
    p.update(kernels.waste_solutions(composting=comp, E0=E0, scenario_name="Composting" + suffix))
    p.update(kernels.waste_solutions(food_loss_waste=flw, E0=E0, scenario_name="Food loss & waste" + suffix))
    # Diets claims FLW CH4 and ecosystem protection claims FLW AFOLU CO2,
    # so neither is double counted in the stack
    waste_stack = kernels.waste_solutions(capture=fc, biocovers=fb, composting=comp,
                                          food_loss_waste=flw, E0=E0, scenario_name="w",
                                          diets_claims_flw_ch4=True,
                                          ecosystems_claim_flw_afolu=True)["w"]
    p["All waste" + suffix] = waste_stack

    # SLOW: diets -- get_input, then herd-adjustment lag kernel
    d = explorer.get_input("Improve Diets", S_curve, add["Improve Diets"], start,
                           land=True, function_kwargs=speed["slow"],
                           scenario_name="Improve Diets")["Improve Diets"]
    # Diets is treated as enabling-only: it lowers the food system's GHG intensity
    # (fewer ruminants -> less enteric/manure CH4) directly. Remove the AFOLU CO2
    # emissions avoided -- we account for those in reduced deforestation
    d = {sp: ts for sp, ts in d.items() if sp != "CO2 AFOLU"}
    h = kernels.exp_kernel(n, HERD_LAG_YR)
    diets = {sp: np.convolve(ts, h)[:n] for sp, ts in d.items()}
    p["Improve Diets" + suffix] = diets

    # speed-class subtotals and the full stack (waste interactions kept)
    p["Fast brakes" + suffix] = stack(p["Manage Oil & Gas Methane" + suffix],
                                      p["Manage Coal Mine Methane" + suffix])
    p["All brakes" + suffix] = stack(p["Fast brakes" + suffix], waste_stack, diets)
    return p


def _as_list(x, allowed, what):
    """Normalize a str/list kwarg ("forests" -> "forest") and check it against allowed."""
    items = [s.lower() for s in ([x] if isinstance(x, str) else x)]
    items = [s[:-1] if s not in allowed and s[:-1] in allowed else s for s in items]
    bad = [s for s in items if s not in allowed]
    if bad:
        raise ValueError(f"unknown {what} {bad}; choose from {list(allowed)}")
    return items


def build_land_brakes(start=2026, ecosystem=None, climate_zone=None, adoption_speed="fast",
                      speed=None, add=None, suffix=""):
    """
    Protect X land/ocean brakes: one perturbation dict per ecosystem and climate
    zone, a subtotal per ecosystem ("All forests", ...), and "All land protection".

    Parameters:
        start: year protection begins
        ecosystem: "forest", "peatland", "grassland", "seaweed", or a list of them;
            defaults to all four
        climate_zone: "boreal", "subtropical", "temperate", "tropical", or a list;
            defaults to all four. Seaweed has no climate zones (its global ceiling
            is split 50/50 across macroalgae types), so it is left out when
            climate_zone is given, unless named explicitly in ecosystem
        adoption_speed: speed class ("fast", "medium", "slow") for every ecosystem
        speed: output of make_speeds; defaults to make_speeds(3, 7, 12)
        add: output of adoption_ceilings; defaults to adoption_ceilings("high")
        suffix: appended to every scenario name
    """
    if speed is None:
        speed = make_speeds(3, 7, 12)
    if add is None:
        add = adoption_ceilings("high")
    zones = CLIMATE_ZONES if climate_zone is None else _as_list(climate_zone, CLIMATE_ZONES, "climate_zone")
    if ecosystem is None:
        ecosystems = [e for e in LAND_ECOSYSTEMS if climate_zone is None or e != "seaweed"]
    else:
        ecosystems = _as_list(ecosystem, LAND_ECOSYSTEMS, "ecosystem")

    p = {}
    subtotals = []
    for eco in ecosystems:
        fn, name, total_key = LAND_ECOSYSTEMS[eco]
        if eco == "seaweed":
            seaweed_hectares = add[name]
            macroalgae_types = kernels.read_seaweed_data()["macroalgae_type"].tolist()
            groups = [(m, f"{name}: {m}", seaweed_hectares / len(macroalgae_types))
                      for m in macroalgae_types]
        else:
            groups = [(z, f"{name}: {z.capitalize()}", add[f"{name}: {z.capitalize()}"])
                      for z in zones]
        keys = []
        for group_value, key, hectares in groups:
            p.update(kernels.get_land_input(fn, group_value, S_curve, hectares, start,
                                            function_kwargs=speed[adoption_speed],
                                            ecosystem_kwargs={}, scenario_name=key + suffix))
            keys.append(key + suffix)
        p[total_key + suffix] = stack(*[p[k] for k in keys])
        subtotals.append(total_key + suffix)

    p["All land protection" + suffix] = stack(*[p[k] for k in subtotals])
    return p
