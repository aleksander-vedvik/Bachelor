# Preparation of data

The files in this directory are used to prepare the datasets for training. A short description of all files will be given here:

- `prepare_data.py`: Reads annotation files in .json and .xml format and provides a dictionary with a common interface for the other scripts.
- `preview_data.py`: Gives a visual representation of the data being prepared in `prepare_data.py`. Used to validate the annotations and images.
- `generate_records.py`: Reads the dictionary from `prepare_data.py` and creates .tfrecords files in the "annotations" directory, which can be used to train and evaluate the models from TensorFlowAPI.
- `generate_yolo.py`: Reads the dictionary from `prepare_data.py` and creates annotation files and images in the "yolov5" directory, which can be used to train and evaluate the yolov5 model.

