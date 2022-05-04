# Pre-trained models from TensorFlowAPI

Donwload an put the pre-trained models into folders. The ones used in the thesis are as follows:
- efficientdet_d0_coco17_tpu-32
- ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
- faster_rcnn_resnet101_v1_640x640_coco17_tpu-8

The `pipeline.config` files in the "ROOT/training/models/*MODEL_NAME*" directories have beeen customized for the models above. Change the fine-tune checkpoint in the `pipeline.config` files if other pre-trained models are used.