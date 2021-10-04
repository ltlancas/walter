# popdet-calc
This is a code to calculate the number density of stars detected in a given observation aiming to resolve a stellar population. As the upcoming Nancy Grace Roman Space Telscope (Roman) will have excellent capabilities for doing these sorts of observations, all of the code laid out here is applied to the specific case of Roman. The details of the mathematical formalism used in this repository are described in the paper accompanying the code, which is available at (...).

The main points of interest to the user are as follows. A walkthrough of how to use th code is available in the iPython notebook `walkthrough.ipynb` with a place to put in your own code at the end of the file. An application of the code to the observation of an example galaxy halo is given in `observation.ipynb`. Finally, code which allows the user to calculate the required exposure time to a given feature of a stellar population (such as its Red Giant Branch) is given in `time_to_feature.ipynb`.

Data that is used in the above mentioned notebooks is stored in the `data` directory. All of this data has been pre-computed using code stored in the `calculation_code` directory. However, in order to use the code `calculation_code` to perform these calculations, the user must download they're own isochrones and also install `Cython`, which is used to speed up some of the calculations. This is not needed for any simple applications of this code to the case of Roman.
