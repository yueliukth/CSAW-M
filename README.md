# CSAW-M
This repository contains code for _CSAW-M: An Ordinal Classification Dataset for Benchmarking Mammographic Masking of Cancer_.
_---- This This repo is under construction and will be updated soon!----_

Source code for training models to estimate the mammographic masking level are made available here.

### Training
In order to train a model, please refer to `scripts/train.sh` where we have prepared commands and arguments to train a model.

### Evaluation
In order to evaluate a trained model, please refer to `scripts/eval.sh` with example commands and arguments to evaluate a model.

### Important arguments
`model_name`: specifies the model name, which will then be used for saving/loading checkpoints  
`loss_type`: defines which loss type to train the model with. It could be either `one_hot` which means training the model in a multi-class setup under usual cross entropy loss, or `multi_hot` which means training the model in a multi-label setup using multi-hot encoding (defined for ordinal labels). Please refer to paper for more details.  
`img_size`: specifies the image size to train the model with.  

