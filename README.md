# Drawdown Impacts
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
5. run `jupyter lab`
6. navigate to the `notebooks` directory and run
*`Examples.ipynb`  for  basic examples
* `TLC.ipynb` for an example of the time/location/cobenefits framework
* `Aerosols.ipynb` for an example making the case for emergency brakes 
