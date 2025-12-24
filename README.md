# Grokking-InfoPlane

This project explores the training dynamics during Grokking on Information Plane [1] of a 3-layer MLP. The Mutual Information (MI) estimation method is from *“Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression.”* [2] (using KSG from https://github.com/VanessB/mutinfo) and the Grokking experiment is adopted from *“Omnigrok: Grokking Beyond Algorithmic Data.”* [3].

[Work in progress]

[TODO] As per the theory proposed by Shwartz-Ziv and Tishby [2], the training can be divided into two separate phases: empirical error minimization (ERM), basically fitting, and representation compression. The results show that the dynamics of grokking in the Information Plane can be split into 3 phases: (1) the overfitting phase, where both MI values decrease. (2) the empirical error minimization phase, where both MI values increase, as well as test accuracy. (3) The representation compression phase, where MI with the input decreases.


## Setup

1) prepare and activate a virtual environment
2) run `pip install -r ./requirements.txt` in the project folder

## Running the code

```bash
# Run the training
python train_mnist.py

# The code uses wandb with hydra so you can specify configs
python train_mnist.py --config-path configs/custom_folder --config-name custom_config.yaml

# and certain params 
python train_mnist.py --config-path configs/custom_folder --config-name custom_config.yaml  model.initialization_scale=6

# You can actually get elaborate like that, but it gets ugly (for win10 cmd)
python train_mnist.py --config-path configs/ --config-name default.yaml ^
    project_name=custom_project_name ^
    custom_run_name=scale_${model.initialization_scale}-seed_${seed}-steps_${train.optimization_steps}-wd_${train.weight_decay}^
    model.initialization_scale=6 ^
    seed=123 ^
    train.optimization_steps=200000 ^
    train.weight_decay=0
```

`plot.ipynb` - notebook for plotting figures from this readme.

`plot_play.ipynb` - notebook with figures from my previous experiments and sweeps (over init_scale, wd, lr).

## Experiments
TODO

## Discussion
TODO

TODO sribnyak/mutinfo -> VanessB/mutinfo (in requirements.txt)

## References
[1] R. Shwartz-Ziv and N. Tishby, “Opening the Black Box of Deep Neural Networks via Information.” arXiv, Apr. 29, 2017. doi: 10.48550/arXiv.1703.00810.

[2] I. Butakov, A. Tolmachev, S. Malanchuk, A. Neopryatnaya, A. Frolov, and K. Andreev, “Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression.” arXiv, May 13, 2023. doi: 10.48550/arXiv.2305.08013.

[3] Z. Liu, E. J. Michaud, and M. Tegmark, “Omnigrok: Grokking Beyond Algorithmic Data.” arXiv, Mar. 23, 2023. doi: 10.48550/arXiv.2210.01117.
