# SurfTomo_CNN

**Reference: Hu et al., (2019) Using deep learning to derive shear wave velocity models from surface wave dispersion [[submitted to SRL]](https://jhsa26.github.io/book/Mypaper/Hu_etal2019SRL.pdf)**

This repository is used to store scripts and an example of surface tomography given real data set

The dataset and scripts are the part of the paper (section 3.1) for the continental China case.

# Configuration 

- You should install Pytorch 0.4 version and Anaconda3.4 and tensorboardX and basemap (for plotting Vs). 

`conda install pytorch=0.4.0 cuda90 -c pytorch; pip install tensorboardX;`

For basemap installation see https://matplotlib.org/basemap/users/installing.html

All python scripts are py3.

# unzip the dataset

`cat Dataset.tar.gz.* | tar -zxv `

Training dataset: USA type (~7000 1-D Vs models from the USA (Shen et al., 2013)); USA-Tibet type (USA type + ~640 Tibet models)

Test dataset:    ~4000 pairs of disperion images associated with phase and group velocity (8-50s). (Shen et al., 2016)

# Two tests

- `scriptsUSA`: using only USA type as a training dataset to train and then using test dataset to predict 1-D Vs models

- `scriptsUSATibet`: using USA type plus ~640 Tibet models as a training dataset and then using test dataset to predict 1-D Vs models

# How to run

For both tests, if you train again, run `Main_train.sh`. If you test, run `Main_test.sh`

The trained model at 600th epoch is included in `model_para`


# Plot vs

- `PlotVs` is used to make a comparison between cnn-based Vs models and results of Shen et al. (2016)

It would read `./scriptsUSA/vs_cnn and ./scriptsUSATibet/vs_cnn` as well as `./PlotVs/data/vs_sws_China` to plot


