# Calibration

This is a simple implementation of temperature scaling to a neural network for leukocyte images detection. The full dataset can be found here: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958

To split the dataset into the 5 folds the network has been trained on, you can refer to the text files in the data folder.
If you want to re-create the plots in the results folder, just run `predict_leukocyte_class.py`

## Folders

* code: Here you can find all four python modules.
* data: Contains weight files and text files that list the images in each fold.
* results: Regroups the reliability diagrams before and after temperature scaling for each fold.

## References
* Matek, C., Schwarz, S., Spiekermann, K., & Marr, C. (2019). Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. Nature Machine Intelligence, 1(11), 538-544.

* Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017, July). On calibration of modern neural networks. In International Conference on Machine Learning (pp. 1321-1330). PMLR.