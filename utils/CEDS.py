import xarray as xr
from pathlib import Path
import glob,os,sys
import datetime
import numpy as np
import pandas as pd
from fair.structure.units import desired_emissions_units

_DATA_DIR = Path(__file__).parent.parent / "data"
_CEDS_DIR=_DATA_DIR / "emissions/CEDS_v_2024_04_01_aggregate"
allceds=glob.glob(str(_CEDS_DIR)+ "/*csv")
_CEDS_species=np.unique([x.split("/")[-1].split("_")[0] for x in allceds])


def prefix_to_num(p):
    pdict={"K":1.e3,
           "M":1.e6,
           "G":1.e9,
           "T":1.e12}
    return pdict[p]

def CEDS_name_to_FAIR(specie,co2source="FFI"):
    #non-methane VOCs in CEDS are just VOCs in FAIR
    if specie == "NMVOC":
        fair_name="VOC"
    # Fair specifies sulfur emissions
    elif specie == "SO2":
        fair_name = "Sulfur"
    elif specie == "CO2":
        if co2source in ["AFOLU","FFI"]:
            fair_name = "CO2 "+co2source
    else:
        fair_name=specie
    return str(fair_name)

def CEDS_to_FAIR(X,specie,co2source="FFI"):
    #non-methane VOCs in CEDS are just VOCs in FAIR
    if specie == "NMVOC":
        fair_name="VOC"
    # Fair specifies sulfur emissions
    elif specie == "SO2":
        fair_name = "Sulfur"
    elif specie == "CO2":
        if co2source in ["AFOLU","FFI"]:
            fair_name = "CO2 "+co2source
        else:
            raise TypeError("co2source must be AFOLU or FFI")
    else:
        fair_name = specie
    csvfile=pd.read_csv(glob.glob(str(_CEDS_DIR)+"/*"+specie+"*_global_emissions_by_sector_v*")[0]) 
    CEDSunits=np.unique(csvfile.units)[0]
    FAIR_units = desired_emissions_units.get(fair_name)
    CEDSprefix=prefix_to_num(CEDSunits[0].upper())
    FAIRprefix=prefix_to_num(FAIR_units[0].upper())
    fac=CEDSprefix/FAIRprefix
    return {fair_name:fac*X}

_COUNTRIES=np.array(['abw', 'afg', 'ago', 'alb', 'are', 'arg', 'arm', 'asm', 'atg',
       'aus', 'aut', 'aze', 'bdi', 'bel', 'ben', 'bfa', 'bgd', 'bgr',
       'bhr', 'bhs', 'bih', 'blr', 'blz', 'bmu', 'bol', 'bra', 'brb',
       'brn', 'btn', 'bwa', 'caf', 'can', 'che', 'chl', 'chn', 'civ',
       'cmr', 'cod', 'cog', 'cok', 'col', 'com', 'cpv', 'cri', 'cub',
       'cuw', 'cym', 'cyp', 'cze', 'deu', 'dji', 'dma', 'dnk', 'dom',
       'dza', 'ecu', 'egy', 'eri', 'esh', 'esp', 'est', 'eth', 'fin',
       'fji', 'flk', 'fra', 'fro', 'fsm', 'gab', 'gbr', 'geo', 'gha',
       'gib', 'gin', 'global', 'glp', 'gmb', 'gnb', 'gnq', 'grc', 'grd',
       'grl', 'gtm', 'guf', 'gum', 'guy', 'hkg', 'hnd', 'hrv', 'hti',
       'hun', 'idn', 'ind', 'irl', 'irn', 'irq', 'isl', 'isr', 'ita',
       'jam', 'jor', 'jpn', 'kaz', 'ken', 'kgz', 'khm', 'kir', 'kna',
       'kor', 'kwt', 'lao', 'lbn', 'lbr', 'lby', 'lca', 'lie', 'lka',
       'lso', 'ltu', 'lux', 'lva', 'mac', 'mar', 'mda', 'mdg', 'mdv',
       'mex', 'mhl', 'mkd', 'mli', 'mlt', 'mmr', 'mne', 'mng', 'moz',
       'mrt', 'msr', 'mtq', 'mus', 'mwi', 'mys', 'nam', 'ncl', 'ner',
       'nga', 'nic', 'niu', 'nld', 'nor', 'npl', 'nzl', 'omn', 'pak',
       'pan', 'per', 'phl', 'plw', 'png', 'pol', 'pri', 'prk', 'prt',
       'pry', 'pyf', 'qat', 'reu', 'rou', 'rus', 'rwa', 'sau', 'sdn',
       'sen', 'sgp', 'slb', 'sle', 'slv', 'som', 'spm', 'srb',
       'srb (kosovo)', 'ssd', 'stp', 'sur', 'svk', 'svn', 'swe', 'swz',
       'sxm', 'syc', 'syr', 'tca', 'tcd', 'tgo', 'tha', 'tjk', 'tkl',
       'tkm', 'tls', 'ton', 'tto', 'tun', 'tur', 'twn', 'tza', 'uga',
       'ukr', 'ury', 'usa', 'uzb', 'vct', 'ven', 'vgb', 'vir', 'vnm',
       'vut', 'wlf', 'wsm', 'yem', 'zaf', 'zmb', 'zwe'], dtype=object)

_FUELS=np.array(['biomass', 'brown_coal', 'coal_coke', 'diesel_oil', 'hard_coal',
       'heavy_oil', 'light_oil', 'natural_gas'], dtype=object)
allceds=glob.glob(str(_CEDS_DIR)+ "/*csv")
_CEDS_species=np.unique([x.split("/")[-1].split("_")[0] for x in allceds])
_CEDS_species


def get_electric_sector(specie,disaggregate=False,by="fuel"):
    """
    Read in CEDS data for a particular specie of pollutant
    https://zenodo.org/records/10904361
    """
    if by == "fuel":
        sectorfname=glob.glob(str(_CEDS_DIR)+ "/*"+specie+"*by_sector_fuel*")[0]
    elif by == "country":
        sectorfname=glob.glob(str(_CEDS_DIR)+ "/*"+specie+"*by_country_sector*")[0]
    else:
        raise TypeError("by must be one of [fuel,country]")
    sectordf=pd.read_csv(sectorfname)
    idx=np.min(np.where([x.find("X")==0 for x in sectordf.columns])[0])
    starttime=sectordf.columns[idx].split("X")[-1]
    
    electricity_sectors=['1A1a_Electricity-autoproducer', '1A1a_Electricity-public','1A1a_Heat-production']
    
    xrdataarrays={}
    
    for esector in electricity_sectors:
       
        secdata=sectordf[sectordf.sector == esector]
        units=secdata.units.unique()[0]
        
        subsets=getattr(secdata,by).values

        mydata=np.stack([
            secdata[getattr(secdata,by)==ss].iloc[:,idx:].values.flatten()
            for ss in subsets
        ])

        lastyear=secdata.columns[-1].split("X")[1]
        
        time_axis = pd.date_range(start=starttime, end=lastyear, freq='YS')
        coords={by:subsets,
                "time":time_axis
               }
        attrs=dict(source=sectorfname,units=units,time_created=datetime.date.today())
        label=esector.split("1A1a_")[1].split("-")[1]
        xrdataarrays[label]=xr.DataArray(mydata,coords=coords,attrs=attrs)
    PowerSector=xr.Dataset(xrdataarrays) 
    if disaggregate:
        return PowerSector
    else:
        return PowerSector.public+PowerSector.autoproducer + PowerSector.production

def get_generation():
    consumption_fname= _DATA_DIR / "consumption/yearly_full_release_long_format.csv"
    df=pd.read_csv(consumption_fname)
    condition=np.logical_and(df.Area=="World",df.Category=="Electricity generation")
    condition = np.logical_and(condition,df.Variable == "Total Generation")
    years=df[condition].Year.values
    coords={"time":pd.date_range(start=str(years[0]), end=str(years[-1]), freq='YS')}
    generation=xr.DataArray(data=df[condition].Value.values,coords=coords)
    return generation

def electricity_all_species(ss):
    """
    Get the total power sector emissions by subset
    ss : subset. Must be a country or a fuel
    """
    if ss in _FUELS:
        by = "fuel"
    elif ss in _COUNTRIES:
        by = "country"
    else:
        raise TypeError("by must be in [country,fuel]")

    electricity_sectors = ['1A1a_Electricity-autoproducer', '1A1a_Electricity-public', '1A1a_Heat-production']
    d = {}
    for specie in _CEDS_species:
        if by == "fuel":
            sectorfname = glob.glob(str(_CEDS_DIR) + "/*" + specie + "*by_sector_fuel*")[0]
        else:
            sectorfname = glob.glob(str(_CEDS_DIR) + "/*" + specie + "*by_country_sector*")[0]

        sectordf = pd.read_csv(sectorfname)
        idx = np.min(np.where([x.find("X") == 0 for x in sectordf.columns])[0])
        starttime = sectordf.columns[idx].split("X")[-1]
        lastyear = sectordf.columns[-1].split("X")[1]
        time_axis = pd.date_range(start=starttime, end=lastyear, freq='YS')

        subset_df = sectordf[sectordf.sector.isin(electricity_sectors) & (getattr(sectordf, by) == ss)]

        total = np.zeros(len(time_axis))
        for esector in electricity_sectors:
            row = subset_df[subset_df.sector == esector]
            if len(row):
                total = total + row.iloc[0, idx:].values.astype(float)

        d[specie] = xr.DataArray(total, coords={"time": time_axis})
    return d

def get_global_by_sector(specie):
    searchstring=f"/*{specie}*global_emissions_by_sector_v*"
    df=pd.read_csv(glob.glob(str(_CEDS_DIR)+ searchstring)[0])
    sectors=df.sector.values
    idx=np.min(np.where([x.find("X")==0 for x in df.columns])[0])
    starttime=int(df.columns[idx].split("X")[-1])
    endtime=int(df.columns[-1].split("X")[-1])
    data=df.iloc[:,idx:].values
    timeax=np.arange(starttime,endtime+1)
    coords=dict(sector=sectors,time=timeax)
    da=xr.DataArray(data=data,coords=coords)
    return da
