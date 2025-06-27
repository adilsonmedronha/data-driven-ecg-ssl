# Exploring Data-Oriented Strategies for Training ECG Self-Supervised Models
---
Pytorch official implementation for Exploring Data-Oriented Strategies for Training ECG Self-Supervised Models


The electrocardiogram (ECG) is a non-stationary signal used to assess heart health and to detect systemic conditions such as mental stress and drug toxicity. However, labeling ECG data is notably costly and can limit the applicability of supervised deep neural networks in tasks involving this physiological signal. Self-supervised learning (SSL) is a two-stage approach that has gained attention in this scenario due to its labeling efficiency and knowledge transfer capabilities in various data modalities. However, recent studies describe the difficulties in transferring knowledge across different time series datasets. This work investigates several data-oriented strategies for pretraining self-supervised learning models in a cross-dataset setting to address a challenging ECG classification task. We introduce a data-driven regularization approach that perturbs the pretrained model using signals from multiple domains. We evaluated three well-established SSL models as backbones with different classification architectures across over 100 experiments, achieving superior performance to traditional SSL methods and a state-of-the-art supervised deep learning classification model.



## Setup
We provide a conda environment file: `environment.yml`. To create the environment, run:

```bash
conda env create -f environment.yml # create the conda environment
conda activate ecg-ssl # activate the environment
pip install -e . # install the packag`
```

**Project Structure**
```
.
├── anaylsis.ipynb
├── Configs
├── Dataset
├── finetuning.py
├── main_supervised.py
├── mine_utils.py
├── models
├── README.md
├── requirements.yml
├── Results
├── run_S2V_and_downstream.sh
├── run.sh
├── run_supervised_models.sh
├── ssl_pretraining.py
├── summary
├── utils
└── wandb
```
## Results
![Alt text](assets/table.png)

### Embeddings

![Alt text](assets/TS2VEC_umap_train_comparison.jpg)
![Alt text](assets/TSTCC_umap_train_comparison.jpg)


# Cite
Comming soon!
