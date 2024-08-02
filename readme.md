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

## 2.2 Weights
Download the pretrained weights from [here](https://drive.google.com/drive/u/3/folders/1tK9ECxINYIRlpNZKV9hv_hU61NiolkRC).


Move them into folder ```weights```
> 💡 There are 3 model weights since we use them to get ensemble results.

The folder structure will like this:
```
YOUR_PATH/AUTOMATED-CROP-DISEASE-DIAGNOSIS-FROM-HYPERSPECTRAL-IMAGERY-3RD
├─archive             <- *We don't use this*
│  ├─train
│  │  ├─Health
│  │  ├─Other
│  │  └─Rust
│  └─val
│      └─val
├─ICPR01              <- *We use this*
│  └─kaggle
│      ├─1
│      ├─2
│      └─evaluation
├─models
│  ├─ensemble_classifier.py
│  ├─RCAN.py
│  ├─SSAM_ConvNeXt.py
│  └─SSAM_SWIN.py
├─utils
│  └─dataset.py
└─weights             <- *Put 3 weights into this folder*
   ├─RCAN.pth
   ├─SSAM_ConvNeXt.pt
   └─SSAM_SWIN.pth
```
# 3. Predict
You can use following command to predict the data in ```ICPR01/kaggle/evaluation``` and generate the submission csv file:

```python
python ensemble_predict.py
```
After predicted, the result submission file will be saved in ```ensemble_cls_result/submission_Ensemble.csv```.

The folder structure will like this:
```
YOUR_PATH/AUTOMATED-CROP-DISEASE-DIAGNOSIS-FROM-HYPERSPECTRAL-IMAGERY-3RD
├─archive
│  ├─train
│  │  ├─Health
│  │  ├─Other
│  │  └─Rust
│  └─val
│      └─val
├─ensemble_cls_result        <-*Result folder*
│  ├─submission_Ensemble.csv
│  └─tensorboard_file
├─ICPR01
│  └─kaggle
│      ├─1
│      ├─2
│      └─evaluation
├─models
│  ├─ensemble_classifier.py
│  ├─RCAN.py
│  ├─SSAM_ConvNeXt.py
│  └─SSAM_SWIN.py
├─utils
│  └─dataset.py
└─weights
   ├─RCAN.pth
   ├─SSAM_ConvNeXt.pt
   └─SSAM_SWIN.pth
```
# 4. Visualization
In addition, a TensorBoard file will be saved in for visualizing the entire ensemble model after predicted.

You can use following command to see the visualizing result:
```bash
tensorboard --logdir ensemble_cls_result
```

