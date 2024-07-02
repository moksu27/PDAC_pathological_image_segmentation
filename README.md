# PDAC_Pathological_Image_Segmentation

## This code aims to segment and visualize the Pancreatic Ductal Adenocarcinoma (PDAC) region in pathological images using various segmentation deep learning models.

## Paper
doi: 10.3389/fonc.2024.1346237

## Introduction
The available deep learning models for this project are as follows:
1. Unet
2. FPN
3. DeepLabV3+
4. PSPNet

The encoders of these models utilize the ImageNet pretrained ResNet18 model to extract features. Our model uses the Distributed Data Parallel (DDP) approach for training.

## Getting Started
### Model Train & Test
It is recommended to specify the data path and adjust the model parameters in the config.yaml file before use.

For training the model, the input data should be in the form of patch data.

It is recommended to use the [QuPath](https://qupath.github.io/) program to generate patch data for pathological images. Please refer to the QuPath_WSI_to_Patch.groovy file for generating patch data using QuPath.

### Visualization
<img width="735" alt="low_resolution" src="https://github.com/moksu27/PDAC_pathological_image_segmentation/assets/108471861/66c87cad-fa7c-46ad-9f35-db85ad201d52">

For visualizing the input data, the coordinates of each patch data are required.

It is recommended to use the [PyHIST](https://github.com/manuel-munoz-aguirre/PyHIST) library to generate patch data and then use it for visualization.

The required data files are
1. Tile images
2. TSV file containing tile image coordinates
3. original image SVS file

## Data
you can use image and mask data in this URL: https://docs.google.com/uc?export=download&id=1FwwYqDmIvuT6XzUTpBJha4z90-UfywHm

## Usage
### Model Training
Execution Code: train.py --config "Your Config Path" --save_path "Your Save Path"

Output: Tensorboard log, Trained pth file

### Model Test
Execution Code: test.py --config "Your Config Path" --save_path "Your Save Path" --pth_path "Trained pth Path"

Output: Predicted Figures

![figure_109](https://github.com/moksu27/PDAC_pathological_image_segmentation/assets/108471861/dd894192-7432-448c-8b77-89a722a3f07d)

### Visualization
You can refer to visualize/predict_overlay.ipynb file to visualize the segmentation predictions of the AI model on Whole Slide Images.



## 이 프로젝트는 여러가지 segmentation 딥러닝 모델을 이용하여 병리 영상 중 Pancreatic Ductal Adenocarcinoma(PDAC)의 부분을 세그먼테이션 하고 시각화하는 것을 목표로 합니다.

## Paper
doi: 10.3389/fonc.2024.1346237

## Introduction
사용할 수 있는 딥러닝 모델을 다음과 같습니다.

1. Unet
2. FPN
3. DeepLabV3+
4. PSPNet

모델들의 인코더는 기본적으로 ImageNet pretrained resnet18 model을 이용하여 feature를 추출합니다.

우리의 모델은 기본적으로 Distributed Data Parallel(DDP) 방식으로 모델 학습을 진행합니다.

## Getting Started
### Model Train & Test
model parameter 및 data 경로는 config.yaml 파일에 정의하고 있기 때문에 config.yaml 파일에서 data 경로 지정 및 파라미터를 적절히 조정 후 사용을 권장합니다.

model 학습을 위한 input data의 경우 patch data이어야만 하며 pathological image에 대한 patch data는 [QuPath](https://qupath.github.io/) program을 이용하여 생성하는 것을 권장합니다.

QuPath program을 이용하여 patch data 생성 시 QuPath_WSI_to_Patch.groovy 파일을 참고하길 바랍니다.

### Visualization
<img width="735" alt="low_resolution" src="https://github.com/moksu27/PDAC_pathological_image_segmentation/assets/108471861/66c87cad-fa7c-46ad-9f35-db85ad201d52">

Visualizatino input data의 경우 각 패치 데이터의 좌표 값이 필요하기 때문에 [PyHIST](https://github.com/manuel-munoz-aguirre/PyHIST) 라이브러리를 사용하여 패치 데이터 생성 후 사용하시길 권장합니다.

필요 data file
1. Tile images
2. TSV file (tile image 좌표 값)
3. 원본 이미지 svs file

## Data
URL을 통해 이미지와 마스크 데이터를 이용할 수 있습니다: https://docs.google.com/uc?export=download&id=1FwwYqDmIvuT6XzUTpBJha4z90-UfywHm

## Usage
### Model Training
실행 코드: train.py --config "Your Config Path" --save_path "Your Save Path"

결과물: Tensorboard log, 학습된 pth file

### Model Test
실행 코드: test.py --config "Your Config Path" --save_path "Your Save Path" --pth_path "Trained pth Path"

결과물: Predicted Figures

![figure_109](https://github.com/moksu27/PDAC_pathological_image_segmentation/assets/108471861/dd894192-7432-448c-8b77-89a722a3f07d)

### Visualization
visualize/predict_overlay.ipynb 파일을 참고하여 AI 모델의 Segmentation 예측을 Whole Slide Image로 시각화 할 수 있습니다.
