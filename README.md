# Calibration

This is a simple implementation of temperature scaling to a neural network for leukocyte images detection. The full dataset can be found here: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958

If you want to re-create the plots in the results folder, just run `predict_leukocyte_class.py`

To see temperature scaling in action, select the npy file for the fold of your choice in `temp_scaling.py` and simply run it.

The code for `temp_scaling.py` was taken from https://github.com/markdtw/temperature-scaling-tensorflow, all credit goes to its author .

## References
* Matek, C., Schwarz, S., Spiekermann, K., & Marr, C. (2019). Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. Nature Machine Intelligence, 1(11), 538-544.

* Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017, July). On calibration of modern neural networks. In International Conference on Machine Learning (pp. 1321-1330). PMLR.