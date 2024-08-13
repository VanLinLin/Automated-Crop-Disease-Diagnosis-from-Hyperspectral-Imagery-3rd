# Overview
We are NCKU_ACVLAB. This is our implemented solution for "Automated Crop Disease Diagnosis from Hyperspectral Imagery 3rd" @ ICPR 2024. We achieved a score tied for first place with five other teams.

# 1. Environment
Run the prepare.sh to auto create the virtual environment:
```bash
bash -i prepare_env.sh
```

> ⚠️ The `i` in ``bash -i prepare_env.sh`` is necessary.

# 2. Download dataset & weights
## 2.1 Dataset
Download the dataset from [kaggle](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2024/data).

Then unzip the ```beyond-visible-spectrum-ai-for-agriculture-2024.zip```.


> ⚠️ After unzip, you will have ```acrhive``` and ```ICPR01``` folders, they are both datasets. We only train and evaluation on ICPR01 dataset since the ```archive``` dataset is the old version one.


The folder structure will like this:
```
YOUR_PATH\AUTOMATED-CROP-DISEASE-DIAGNOSIS-FROM-HYPERSPECTRAL-IMAGERY-3RD
└─beyond-visible-spectrum-ai-for-agriculture-2024  <-*Dataset*
  ├─archive
  │  ├─train
  │  │  ├─Health
  │  │  ├─Other
  │  │  └─Rust
  │  └─val
  │      └─val
  └─ICPR01
      └─kaggle
          ├─1
          ├─2
          └─evaluation
```


# 3. Predict
Run the ```ML_methods.ipynb``` can generate the submission files, will be saved in folder ```results```.

```
D:\SIDE_PROJECT\AUTOMATED-CROP-DISEASE-DIAGNOSIS-FROM-HYPERSPECTRAL-IMAGERY-3RD
├─beyond-visible-spectrum-ai-for-agriculture-2024
│  ├─archive
│  │  ├─train
│  │  │  ├─Health
│  │  │  ├─Other
│  │  │  └─Rust
│  │  └─val
│  │      └─val
│  └─ICPR01
│      └─kaggle
│          ├─1
│          ├─2
│          └─evaluation
├─results                   <- *Submissions files will saved in here*
└─__pycache__
```
