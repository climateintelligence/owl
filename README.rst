===
owl
===


.. image:: https://img.shields.io/pypi/v/owl.svg
        :target: https://pypi.python.org/pypi/owl

.. image:: https://img.shields.io/travis/climateintelligence/owl.svg
        :target: https://travis-ci.com/climateintelligence/owl

.. image:: https://readthedocs.org/projects/owl/badge/?version=latest
        :target: https://owl.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/license/climateintelligence/owl.svg
    :target: https://github.com/climateintelligence/owl/blob/master/LICENSE.txt
    :alt: GitHub license

.. image:: https://badges.gitter.im/bird-house/birdhouse.svg
    :target: https://gitter.im/bird-house/birdhouse?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
    :alt: Join the chat at https://gitter.im/bird-house/birdhouse

Owl (the bird)
  *Owl is a bird designed to detect both day-time and night-time temperature extremes, and their drivers. *

Owls are nocturnal creatures with exceptional long-distance vision and the ability to rotate their heads up to 270 degrees. Much like the owl, this prototype can "see" far and wide; it is applicable to all temperature extremes (day and night), and connects the past drivers of extremes to projections of the future.

Specifically, the Owl prototype (1) detects extremes from a given temperature time series input, (2) performs a feature selection on a range of potential predictors and (3) identifies storylines in future projections of these extremes.

Here we describe the three working parts of the Owl:

(1) Heatwave Index Detection
Here, the Owl follows the widely used definition of heatwaves: temperatures exceed the 90th percentile for 3 days or longer (e.g Russo et al., 2015). Heatwaves are typically detected using daily maximum 2m temperature, although minimum temperature can be used for night-time heatwaves. 
The first role of the Owl is to detect the heatwave occurrence in the target time series, achieved by: 
- calculating the 90th percentile (threshold)
- identifying the occurrences of exceedence of the threshold
- outputting daily time series of heatwaves 

The detection code allows the user to choose a range of parameters, such as the climatology period over which the treshold is calculated, the minimum duration of the HWs (e.g. 3 days as above). It also allows for the detection of heatwaves in any (daily) temperature time series, and is demonstrated here for the ERA5 reanalysis (Hersbach et al., 2021) and CMIP6 historical periods and projections. The examples correspond to the area-averaged temperatures over the Po Valley, Italy.

Fig X: HW occurrence over Po Valley 1950-2022 in ERA5.

(2) Feature Selection
The second role of the Owl is to employ a feature selection algorithm to detect the drivers, from a list of potential predictors, of the HW index produced in step 1.  The potential predictors provided in the prototype are area-averages of clusters identfied by a k-means clusters of the following variables: precipitation, 2m temperature, mean sea level pressure, geopotential height, soil moisture, outgoing long-wave radiation, sea ice cover, sea surface temperature. 

The feature selection used here is a machine learning classifier wrapped in an optimization algorithm which detects the optimal combinations of predictors (i.e. those which provide the best skill in recreating the HW time series). The HW target data and predictors are split into training and testing periods. The ML classifier selectes combinations of potential predictors to recreate the test data, providing a measure of skill (i.e. F1-score). Then, the Coral Reef Optimization algorithm  (Salcedo-Sanz et al., 20; Perez-Aracil et al.,) works to select the optimal combination (i.e. to increase the skill).

The output is a list of predictors which contribute to recreating the HW time series, providing information also on the relevant time lags for each selected predictor.

Fig X: Feature Selection: optimal predictors and lags used to recreate HW index in Figure.

(3)

Input: pre-prepared candidate predictors (for each CMIP6 or ERA5) + HW indices from 4.1
CRO (refer to other MS)
Output: Visualization of selected features (evolution of FS, scores and board of selected features)

Projections/Storylines
CMIP6 model driver validation: selection of models which share common drivers with ERA5; returns a list of models for each pair of drivers of the users choice
Requirements: certain number of models…
Storylines: scatter plots / identification of storylines / related HW indices


Documentation
-------------

Learn more about owl in its official documentation at
https://owl.readthedocs.io.

Submit bug reports, questions and feature requests at
https://github.com/climateintelligence/owl/issues

Contributing
------------

You can find information about contributing in our `Developer Guide`_.

Please use bumpversion_ to release a new version.


License
-------

* Free software: Apache Software License 2.0
* Documentation: https://owl.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `bird-house/cookiecutter-birdhouse`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`bird-house/cookiecutter-birdhouse`: https://github.com/bird-house/cookiecutter-birdhouse
.. _`Developer Guide`: https://owl.readthedocs.io/en/latest/dev_guide.html
.. _bumpversion: https://owl.readthedocs.io/en/latest/dev_guide.html#bump-a-new-version
