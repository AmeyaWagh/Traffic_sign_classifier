# Traffic sign classifier 
This repository uses pytorch and pytorch-lightning as the training frameworks.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)

### - ###

![alt text](test/test_50_1.jpg)


### GTSRB dataset:
`https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download?datasetVersionNumber=1`

### Training the network:
```bash
python3 scripts/train.py --config  scripts/config.yaml
```

### Predict
use the notebook provided to predict traffic signs.

```bash
notebooks/predict.ipynb
```