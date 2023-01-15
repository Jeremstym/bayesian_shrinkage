# Bayesian estimation of heavy-tailed densities
This repository to estimate prior and posterior laws  in order to create a density that takes into account heavy-tail distribution and does not shrink high outliers.

We take our main framework from the article : *On Global-Local Shrinkage Priors for Count Data* of Y. Hamura, K. Irie and S. Sugasawa

R code is drawn from : https://github.com/sshonosuke/GLSP-count

Clone the repository with the command:
```bash
git clone https://github.com/Jeremstym/bayesian_shrinkage.git
```
Or download the zip from the [github](https://github.com/Jeremstym/bayesian_shrinkage) and unzip it where you want.

### Create conda enviromnent

You can create a conda environment before installing all packages (fist line if you work on windows, second one on linux)

```bash
conda create --name YOURENV --file conda-osx-64.lock
conda create --name YOURENV --file conda-linux-64.lock
```
Then, yo can activate the environment

```bash
conda activate YOURENV
```

To generate the lock file from `environment.yml` do
```bash
conda-lock -k explicit --conda mamba
```

If you want to try the visualization, you have to use special packages (not in the yml file)

```bash
pip install requests py7zr geopandas openpyxl tqdm s3fs PyYAML xlrd
pip install git+https://github.com/inseefrlab/cartogether
```
