# Li-Ion ML

![Two plots of distributions of compounds based on compositional similarity. The left has each of the lithium containing materials overlaid in black, the right has the identified electrolyte materials from our work overlaid in colour.](img/Figure3.png)

A collection of notebooks in support of the publication A Database of Experimentally Measured Lithium Solid Electrolyte Conductivities Evaluated with Machine Learning. 

To run these notebooks please accept the terms of using the dataset and download the csv from https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html.

Recommended usage is from a clean python virtual environment and installing required packages from the `environment.yml` file. For example, using conda:

```
git clone https://github.com/lrcfmd/LiIonML
cd LiIonMl
conda env create -f environment.yml
conda activate LiIonMl
```

CrabNet models are cloned from the original repository, https://github.com/anthony-wang/CrabNet, with minor modifications to allow early stopping. 

Please feel free to post any problems or queries you encounter as issues on this GitHub page
