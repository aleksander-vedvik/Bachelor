# Bachelor
I will provide a quick guide below on how to get started with this repository.

ROOT or . is defined as the same folder where this readme file is located.


# Installation

## Prerequisite
Install dependencies:

    pip install requirements.txt

## TensorFlowAPI:
1. Download protoc and add to PATH:
    - Download [Protoc](https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip).

    - Add protoc to Windows PATH: "*_PATH TO DIRECTORY_*\protoc\protoc-3.15.6-win64\bin"

2. Clone [this repository](https://github.com/tensorflow/models) into the "./training/tensorflowapi/" folder.

3. Install dependencies:
   
        cd ./training/tensorflowapi/research/slim && pip install -e .

4. Download pre-trained models [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and put them in the "./training/pre-trained-models" folder. The models used in this thesis are: ssd_mobnet, efficientdet and faster_rcnn.


## YOLOv5:
Follow the installation steps provided [here](https://github.com/ultralytics/yolov5) by Ultralytics. Clone the repository into the "./training/yolov5/" folder.
   

# Prepare the data

## Datasets

1. Download the datasets from here.
2. Unzip the datasets into the folder "./data/"

## TensorFlowAPI:

Create .tfrecords by running this command from ROOT:

    cd ./training/scripts
    python generate_records.py


## YOLOv5:

Prepare the training and validation dataset for YOLOv5 by running this command from ROOT:

    cd ./training/scripts
    python generate_yolo.py


# Train

## TensforFlowAPI:
Run this command from ROOT to train a model from TensorFlowAPI:

    python Models/models/research/object_detection/model_main_tf2.py --model_dir=training/models/*MODEL_NAME* --pipeline_config_path=training/models/*MODEL_NAME*/pipeline.config --num_train_steps=*NUMBER_OF_STEPS*

The different models used are: 
- Model name: **ssd_mobnet**, number of steps: **10 000**
- Model name: **efficientdet**, number of steps: **15 000**
- Model name: **faster_rcnn**, number of steps: **20 000**

## YOLOv5:

    python ./yolov5/train.py --img 1920 --batch 4 --epoch 30 --data ./dataset.yaml --cfg ./yolov5/models/yolov5x.yaml --weights ./yolov5x.pt --name ./yolov5x_vehicles --cache



# Eval

## TensforFlowAPI:
Run this command from ROOT to train a model from TensorFlowAPI:

    python Models/models/research/object_detection/model_main_tf2.py --model_dir=training/models/*MODEL_NAME* --pipeline_config_path=training/models/*MODEL_NAME*/pipeline.config --checkpoint_dir=training/models/*MODEL_NAME*

The different models used are: 
- Model name: **ssd_mobnet**
- Model name: **efficientdet**
- Model name: **faster_rcnn**


# Tensorboard

You can use Tensorboard to get info regarding

### train
cd ../train
cd Tensorflow/workspace/models/ssd_mobnet/train
tensorboard --logdir=.

### test
cd ../eval
cd Tensorflow/workspace/models/ssd_mobnet/eval
tensorboard --logdir=.

cd Models/workspace/Detection/TensorFlowAPI/models/ssd_mobnet/eval
cd Models/workspace/Detection/TensorFlowAPI/models/efficientdet/eval
cd Models/workspace/Detection/TensorFlowAPI/models/faster_rcnn/eval


# Test the models on the test dataset:

Run this command from ROOT to test the models:

    python run.py -m *model* -c *checkpoint* -r *scale_factor* -t *tracking_model* -i *image_enhancement_method* -f *save_to_file* -s *number* -p *model*

Description of flags:
- -m: Specify which object detection model to use. 
  - **Options**: yolov5, ssd_mobnet, efficientdet, faster_rcnn
  - **Default**: yolov5
- -c: Specify the checkpoint to use. Only relevant for the models from TensorFlowAPI.
- -r: Resize the input frame. 


# References:

- https://github.com/tensorflow/models

- https://github.com/nicknochnack/TFODCourse