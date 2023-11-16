template ML project
==============================

This is a template for a predictive modeling project. You have some features (X) and want to predict some target (y). This project uses pydra-ml. Flexible routine for creating spec files: features, targets, participants.


First Steps
------------

* Clone Repo
> Clone the repo to your local path
```
git clone git@github.com:maedbhk/template_ML.git`
```

* Activate Virtual Environment
In this project, we're using [**conda**](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for virtual environment and python package management

* To install and activate conda environment, run the following commands in the top-level directory of your cloned repo
```
# navigate to top-level directory of repo
$ cd ../template_ML

# create conda environment using .yml file
$ conda env create -f environment.yml 

# activate the conda virtual environment
$ conda activate template_ML # each virtual env has its own unique name (look inside the environment.yml file)

# install editable package (make sure virtual env is activated) so that you can access `src` packages
$ pip install -e .
```

* Activate jupyter notebook kernel
> To run jupyter notebook using modules installed in virtual env, run the following command in top-level directory of repo
```
# make sure you are in top-level directory of repo
$ cd ../template_ML

# make sure conda env is activated
$ conda activate template_ML

# create kernel (virtual env for jupyter notebook)
$ ipython kernel install --name "template_ML" --user # you can assign any name
```

Notebooks
------------
> To access the jupyter notebook, run the following commands
```
# first, make sure your virtual environment is activated
$ conda activate template_ML

# navigate to the notebooks directory
$ cd ../template_ML/notebooks

# open the jupyter notebook
$ jupyter notebook 1.1.Workflow-Example.ipynb

# once you have opened the notebook, make sure you choose the correct kernel "template_ML" as it contains all of the python packages for this project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml    <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `conda env create -f environment.yml`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Functions to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Functions to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Functions to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   ├── scripts         <- Scripts 
    │   │   ├── make_model.py
    │   │   └── make_specs.py
    │   │   └── make_summary.py
    │   │   └── make_train_test_split.py
    │   │   └── run_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
