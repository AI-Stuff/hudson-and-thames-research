# Research Repo
Contains all the Jupyter Notebooks used in our research.

## Additional Research Repo
BlackArbsCEO has a great repo based on de Prado's research. It covers many of the questions at the back of every chapter and was the first source on Github to do so. It has also been a good source of inspiration for our research.

* [Adv Fin ML Exercises](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises)

## Sample Data
The following [folder](https://github.com/hudson-and-thames/research/tree/master/Sample-Data) contains 2 years sample data on S&P500 Emini Futures, for the period 2015-01-01 to 2017-01-01.

Specifically the following data structures:
* Dollar Bars: Sampled every $70'000
* Volume Bars: Sampled every 28'000 contracts
* Tick Bars: Sampled every 2'800 ticks

Our hope is that the following samples will enable the community to build on the research and contribute to the open source community.

A good place to start for new users is to use the data provided to answer the questions at the back of chapter 2 of Advances in Financial Machine Learning.

## Naming convention:
```yyyy-mm-dd_initials_dash-separated-notebook-description``` Example: 2019-02-22_JFJ_meta-labels.

## Installation on Mac OS X and Ubuntu Linux
Make sure you install the latest version of the Anaconda 3 distribution. To do this you can follow the install and update instructions found on this link: https://www.anaconda.com/download/#mac

### Create a New Conda Environment
From terminal: ```conda create -n <env name> python=3.6 anaconda``` accept all the requests to install.

Now activate the environment with ```source activate <env name>```.

### Install Packages
From Terminal: go to the directory where you have saved the file, example: cd Desktop/research/.

Run the command: ```pip install -r pip_requirements.txt```
