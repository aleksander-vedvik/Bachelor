# Bachelor
I will provide a quick guide below on how to get started with this repository.


# Installation

## TensorFlowAPI
1. Download protoc and add to PATH:
    - Download [Protoc](https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip).

    - Add protoc to PATH: "*_PATH TO DIRECTORY_*\protoc\protoc-3.15.6-win64\bin"

2. Clone [this repository](https://github.com/tensorflow/models) into the "./training/tensorflowapi" folder.

3. Install dependencies:
   
        cd ./training/tensorflowapi/research/slim && pip install -e .

4. Download pre-trained models [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and put them in the "./training/pre-trained-models" folder.


## YOLOv5
Follow the installation steps provided [here](https://github.com/ultralytics/yolov5) by Ultralytics. Clone the repository into the ".training/yolov5" folder.


# Prepare the data


# Generate records from .txt files.
    python Tensorflow/other/generate_records.py

cd C:/Users/Aleks/Documents/Bachelor/Workspace/Models/scripts/training
python generate_records.py

cd C:/Users/Aleks/Documents/Bachelor/Workspace

# Train:
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/ssd_mobnet/pipeline.config --num_train_steps=2000
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/ssd_resnet --pipeline_config_path=Tensorflow/workspace/models/ssd_resnet/pipeline.config --num_train_steps=2000
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/faster_rcnn --pipeline_config_path=Tensorflow/workspace/models/faster_rcnn/pipeline.config --num_train_steps=2000
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/mask_rcnn --pipeline_config_path=Tensorflow/workspace/models/mask_rcnn/pipeline.config --num_train_steps=2000

python Models/models/research/object_detection/model_main_tf2.py --model_dir=Models/workspace/Detection/TensorFlowAPI/models/ssd_mobnet --pipeline_config_path=Models/workspace/Detection/TensorFlowAPI/models/ssd_mobnet/pipeline.config --num_train_steps=10000

python Models/models/research/object_detection/model_main_tf2.py --model_dir=Models/workspace/Detection/TensorFlowAPI/models/efficientdet --pipeline_config_path=Models/workspace/Detection/TensorFlowAPI/models/efficientdet/pipeline.config --num_train_steps=15000

python Models/models/research/object_detection/model_main_tf2.py --model_dir=Models/workspace/Detection/TensorFlowAPI/models/faster_rcnn --pipeline_config_path=Models/workspace/Detection/TensorFlowAPI/models/faster_rcnn/pipeline.config --num_train_steps=20000




# Test/Eval:
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/ssd_mobnet/pipeline.config --checkpoint_dir=Tensorflow/workspace/models/ssd_mobnet
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/ssd_resnet --pipeline_config_path=Tensorflow/workspace/models/ssd_resnet/pipeline.config --checkpoint_dir=Tensorflow/workspace/models/ssd_resnet
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/faster_rcnn --pipeline_config_path=Tensorflow/workspace/models/faster_rcnn/pipeline.config --checkpoint_dir=Tensorflow/workspace/models/faster_rcnn
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/mask_rcnn --pipeline_config_path=Tensorflow/workspace/models/mask_rcnn/pipeline.config --checkpoint_dir=Tensorflow/workspace/models/mask_rcnn

python Models/models/research/object_detection/model_main_tf2.py --model_dir=Models/workspace/Detection/TensorFlowAPI/models/ssd_mobnet --pipeline_config_path=Models/workspace/Detection/TensorFlowAPI/models/ssd_mobnet/pipeline.config --checkpoint_dir=Models/workspace/Detection/TensorFlowAPI/models/ssd_mobnet

python Models/models/research/object_detection/model_main_tf2.py --model_dir=Models/workspace/Detection/TensorFlowAPI/models/efficientdet --pipeline_config_path=Models/workspace/Detection/TensorFlowAPI/models/efficientdet/pipeline.config --checkpoint_dir=Models/workspace/Detection/TensorFlowAPI/models/efficientdet

python Models/models/research/object_detection/model_main_tf2.py --model_dir=Models/workspace/Detection/TensorFlowAPI/models/faster_rcnn --pipeline_config_path=Models/workspace/Detection/TensorFlowAPI/models/faster_rcnn/pipeline.config --checkpoint_dir=Models/workspace/Detection/TensorFlowAPI/models/faster_rcnn


# GUI

cd C:/Users/Aleks/Documents/Bachelor/Workspace/Models/workspace/Detection/TensorFlowAPI/models/ssd_mobnet

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

# Visual display of a single image from trained model using matplotlib
python ./Tensorflow/other/detect_objects.py

# Visual display of a video from trained model using OpenCV
python ./Tensorflow/other/rt_detection.py
python ./Tensorflow/scripts/detection/detect.py -m *model* -c *checkpoint_number*  
# Both arguments are optional.
# models: ssd_mobnet, ssd_resnet, faster_rcnn, mask_rcnn.
# checkpoint_number: Checkpoint number the trained model should use when doing real-time detection.


# Generate training and test sets
python ./Tensorflow/scripts/training/prepare_data.py

# Create YOLOv5 data:
python ./Tensorflow/scripts/training/generate_yolo.py


# TRAIN YOLOv5 MODEL:
python ./yolov5/train.py --img 1920 --batch 4 --epoch 30 --data ./dataset.yaml --cfg ./yolov5/models/yolov5x.yaml --weights ./yolov5x.pt --name ./yolov5x_vehicles --cache



### OBJECT TRACKING
cd C:/Users/Aleks/Documents/Bachelor/Workspace/Models/workspace/Tracking/DeepSort
python object_tracking.py -m *model* -c *checkpoint*



# References:

- https://github.com/tensorflow/models

- https://github.com/nicknochnack/TFODCourse