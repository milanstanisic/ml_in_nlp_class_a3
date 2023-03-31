# Assignment 3 - Neural Networks - Milan Stanisic
**DESCRIPTION:** Assignment 3 for the Machine Learning in NLP class


## Documentation 

`a3_features.py`

This script preprocesses raw text data and creates features out of them that are ready to use by the `a3_model` script. It is composed of two key blocks:
1. Data-reading block (lines 19-57): a loop that iterates over distinct folders and files. 

In the beginning, an empty feature data frame is instantiated, and then the iteration process takes place. Each file is read line-by-line, one line at a time to reduce the memory load. After reading each line, a couple of conditions based on boolean values are checked to assess whether or not a given line is a continuation of a message, an indicator of a new message or system line that is not an actual message. There are three boolean values:
- `new_message`: if `True`, it indicates that a new message has been observed in the same file (there can be multiple messages authored by various people in one file). It is set to `False` when the new message is being parsed to make sure it is read entirely. It is set to `True` only if a line saying `"-----Original Message-----"` is encountered.
- `first_message`: indicates that there was no message recorded for the file yet - it helps ignoring the system lines before reaching the actual message. 
- `wrong_author`: indicates whether or not te message was authored by the author the data are collected of. If `False`, the whole message is ignored until a new message appears. 

After parsing each message, the message is added to the raw data frame along with the author's name as its label. 

2. Encoding block (lines 61-83): this block vectorizes raw text data extracted from the data frame and reduces its number of dimensions to the desired value. For each message in the data frame, it computes the tf-idf vector (it is not optimal since using stop words and grammatical dependencies between them would have been more powerful in author classification but given that the number of dimensions was to be specified by the user, I selected this startegy). Then, it uses the truncated SVD dimensionality reduction method (initially, a PCA was tested but the input tf-idf vectors were too sparse) which reduces the number of dimensions down to the number desired by the user. 

In the final preprocessing step, each entry in the feature data frame is assigned either a `train` or a `test` class, based on the value specified by the user (default: 80% training, 20% test).

Eventually, raw values are removed from the data frame, and only the resulting vectors and their respective labels and subset class remain there. In the last step, the data are saved to a `csv`. 

**NOTE**: the `csv` file is saved in the folder with all author folders. 

`a3_model.py`

This is the model's script. It takes the following arguments: 
- `featurefile`: the name of the file containing feature vectors and labels. Type: `str`
- `hiddensize` (optional): if provided, specifies the size of an optional hidden layer. Type: `int`
- `non_linearity` (optional): if provided, specifies the non-linear activation function of the hidden layer. Type: `str`. Default value: `'sigmoid'`. 

Additional classes: 
- `Dataset`: an object type that keeps data frame data and labels as tensors. 

First, the script performs one-hot encoding fo all labels found in the input feature file.

## Enron Ethics 
