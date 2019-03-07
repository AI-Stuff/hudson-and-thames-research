# Research Repo
Contains all the Jupyter Notebooks used in our research. 

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

## Guidelines:
### General
* Explore only ONE idea.
* Notebooks must be independent, no Notebook-chaining!
* Cells must successfully and correctly execute in order using the "Run all cells" command.
* A notebook should run "as is" if pulled from Git. That is, notebooks may not depend on files on your local computer.

### Style 
* Remove unused or commented out code.
* Remove empty cells.
* Remove imports that are not used in the notebook.
* Code must be PEP8 compliant.
* Notebooks must not contain duplicate code that exists in another notebook. Rather write a function in a common library.

### Figures
* Label figures: include a title, axes descriptions and legends.
* Explain the figure.

### Ease-of-use
* Keep it short (~300 lines of code).
* Cells should be short.
* Have reasonable cell outputs.
* Include an abstract, introduction, conclusion and next steps section at the beginning of your notebook. These sections should be written in such a way that, in 3 months from now, it is still clear what the notebook's purpose is without reading any other notebooks.

### In The Project
* Functions in the notebook must have (at least) a one-liner doc string describing that function.
* Complex functions should have toy examples demonstrating purpose and functionality.
* Pull requests for a notebook must include an update to the solution_document.md file detailing your findings.
