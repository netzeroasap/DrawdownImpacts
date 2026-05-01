# Drawdown Impacts
Converting Drawdown solutions into future climate trajectories

We explore the climate impacts of solutions in the [Explorer](https://drawdown.org/explorer)  using the [FAIR](https://github.com/OMS-NetZero/FAIR) (v2.2.0) simple climate model.

![Global temperature impacts of implementing the Improve Diets
solution slowly (blue) vs immediately (orange)](plots/Drawdown/improve_diets.png)


The code allows the user to specify the
baseline scenario, with a choice between constant emissions or the
CMIP7 high, medium, or low scenarios.  The model is a ``story
machine", taking as imput a hypothetical baseline future and then a
trajectory of emissions avoided per solution.

## how to reproduce
1. clone the repository with `git clone https://github.com/netzeroasap.git`
2. install the environment with `conda env create -f environment.yml`
3. activate environment `conda activate drawdown`
4. Drawdown staff only: Download the service_account.json file in the team Drive: https://drive.google.com/drive/folders/1LLlGB6rMVrfTps2TCDwxMb9u4jtbyMiG
5. Make sure the spreadsheet is shared with explorer-sheet-access@drawdown-solutions-translator.iam.gserviceaccount.com
6. run `jupyter notebook`
7. navigate to the `notebooks` directory and run `Drawdown.ipynb`
