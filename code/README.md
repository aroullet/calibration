# code
`predict_leukocyte_class.py` is the main file used to run the trained network.
 
 `calibration.py` contains all the methods that allow to plot reliability diagrams and compute the calibration metrics.
 
 `residual_network.py` takes care of implementing the ResNeXt CNN.
 
 `temp_scaling.py` implements temperature scaling, its code was taken from https://github.com/markdtw/temperature-scaling-tensorflow, all credit goes to its author.
 
## Reliability Diagrams
To plot the reliability diagram, run `predict_leukocyte_class.py` and change the test_folder to the one of the fold of interest (and don't forget to use the corresponding weights file).

## Temperature scaling
To see temperature scaling in action, first start by editing the last layer of `residual_network.py`, changing the activation from `'softmax'` to `None`.

Run `temp_scaling.py` with the npy file for the fold of your choice and you should get the optimal temperature.

To see how the calibrated network performs, go back to `residual_network.py`, uncomment the last two layers and change the scalar in the lambda layer to the temperature you just got.