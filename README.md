# The Drawdown Worldbuilder
Converting Drawdown solutions into future climate trajectories

We explore the climate impacts of solutions in the [Explorer](https://drawdown.org/explorer)  using the [FAIR](https://github.com/OMS-NetZero/FAIR) (v2.2.0) simple climate model.


The code allows the user to specify the
baseline scenario, with a choice between constant emissions or the
CMIP7 high, medium, or low scenarios.  The model is a ``story
machine", taking as imput a hypothetical baseline future and then a
trajectory of emissions avoided per solution.

## how to reproduce
1. clone the repository with `git clone https://github.com/netzeroasap/DrawdownImpacts.git`
2. install the environment with `conda env create -f environment.yml`
3. activate environment `conda activate drawdown`
4. install this necessary within-repo packages with `pip install -e .`
5. set up access to the solutions explorer data (below)
6. run `jupyter lab`
7. navigate to the `notebooks` directory and run

* `Examples.ipynb` for a basic tutorial
* `TLC.ipynb` for an example of the time/location/cobenefits framework
* `Aerosols.ipynb` for an example making the case for emergency brakes 
* `EmergencyBrakes.ipynb` for how fast solutions act on global temperature, combining deployment, emissions, and atmospheric timescales ([open in Binder](https://mybinder.org/v2/gh/netzeroasap/DrawdownImpacts/HEAD?labpath=notebooks/EmergencyBrakes.ipynb))

## Interface with the solutions explorer
Reduced emissions per solution are read live from the "Drawdown Climate Reduced Emissions Numbers" Google Sheet, which requires the Project Drawdown service account.

1. Download the `service_account.json` file in the team Drive: https://drive.google.com/drive/folders/1LLlGB6rMVrfTps2TCDwxMb9u4jtbyMiG
2. Put it in the top level of this repository (it is gitignored), or at `~/.config/gspread/service_account.json`
3. To work with another spreadsheet, navigate to it in the Google Drive, click "share", and share it with explorer-sheet-access@drawdown-solutions-translator.iam.gserviceaccount.com. Then assign `shname` = the name of that spreadsheet

**DO NOT SHARE THE SERVICE ACCOUNT OUTSIDE PD**

### Without the service account
Trusted collaborators outside PD can run the code from a CSV snapshot of the sheet instead.

* **To make the snapshot** (needs the service account): run `from drawdown import explorer; explorer.export_reduced_emissions_csv()`. This writes the current sheet to `data/emissions/solutions/Drawdown Climate Reduced Emissions Numbers - reduced_emissions.csv`, which you can send directly.
* **To use it:** put the CSV you were sent at that same path. The code uses it automatically when no service account is found, with a warning that it may be out of date.

The CSV is gitignored. Don't commit it.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/netzeroasap/DrawdownImpacts/HEAD?labpath=notebooks/Examples.ipynb)
