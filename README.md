# Diffusion Attention Convolutional Recurrent Neural Network for Traffic Forecasting

![Diffusion Convolutional Recurrent Neural Network](figures/model_architecture.jpg "Model Architecture")

This is a TensorFlow implementation of Diffusion Attention Convolutional Recurrent Neural Network. This code is based on [DCRNN](https://github.com/liyaguang/DCRNN)

## Requirements
- scipy>=0.19.0
- numpy>=1.12.1
- pandas>=0.19.2
- tensorflow>=1.3.0
- pyaml


Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
The traffic data file for Los Angeles, i.e., `METR-LA.h5`, is available at [Google Drive](https://drive.google.com/open?id=1tjf5aXCgUoimvADyxKqb-YUlxP8O46pb), [Baidu Yun](https://pan.baidu.com/s/1rsCq38a9SRyFO1F68tUscA) or [DCRNN](https://github.com/liyaguang/DCRNN), and should be
put into the `data/` folder.

```bash
# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```
The generated train/val/test dataset will be saved at `data/METR-LA/{train,val,test}.npz` or `data/PEMS-BAY/{train,val,test}.npz`.

## Model Training
```bash
python dacrnn_train.py --config_filename=data/model/dcrnn_config.yaml
```
Each epoch takes about 7min~14min with a single GTX 1080 Ti.

