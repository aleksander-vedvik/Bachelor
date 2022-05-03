import os
import io
import sys

PATH_TO_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATH_TO_THIS_FILE + '\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '..\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '..\\tensorflowapi\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '..\\tensorflowapi\\research\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '..\\tensorflowapi\\research\\object_detection')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util

from prepare_data import Prepare

def create_tf_example(example):
    path = example["images_path"]
    image_path = os.path.join(path, '{}'.format(example["filename"]))
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = example["filename"].encode('utf8')
    image_format = b'jpg'

    x1s = []
    x2s = []
    y1s = []
    y2s = []
    classes = []

    for obj in example["objects"]:
        x1s.append(obj['x1'] / width)
        x2s.append(obj['x2'] / width)
        y1s.append(obj['y1'] / height)
        y2s.append(obj['y2'] / height)
        classes.append(obj['class'].encode('utf8'))
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x1s),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x2s),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y1s),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y2s),
        'image/object/class/text': dataset_util.bytes_list_feature(classes),
        'image/object/class/label': dataset_util.int64_list_feature([1]),
    }))
    return tf_example


def main(_):
    image_dir_night = r'..\\..\\data\\Training\\raw_night\\'
    anno_path_night = r'..\\..\\data\\Training\\raw_night\\'
    
    image_dir_raw = r'..\\..\\data\\Training\\raw\\'
    anno_path_raw = r'..\\..\\data\\Training\\raw\\'
    
    image_dir_kitti = r'..\\..\\data\\Training\\kitti\\'
    anno_path_kitti = r'..\\..\\data\\Training\\kitti\\'

    datasets = [ {"dataset": "other1", "images": image_dir_night, "annotations": anno_path_night},
                 {"dataset": "other2", "images": image_dir_raw, "annotations": anno_path_raw},
                 {"dataset": "other3", "images": image_dir_kitti, "annotations": anno_path_kitti}]

    org_path = r'..\\..\\data\\Incidents\\Video'
    for i in range(1, 13):
        image_dir1 = org_path + str(i) + "\\images\\"
        anno_path1 = org_path + str(i) + "\\annotations.json"
        dataset_name = "self_annotated" + str(i)
        datasets.append({"dataset": dataset_name, "images": image_dir1, "annotations": anno_path1})
    
    train_test_distribution = 0.9
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}
    
    preparer = Prepare(train_test_distribution, datasets, classes)
    
    output_path = "../annotations/"

    train_output = output_path + "train.record"    
    writer = tf.python_io.TFRecordWriter(train_output)
    for example in preparer.get_all_train_entries():
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f'Successfully created the TFRecord file: {train_output}')

    test_output = output_path + "test.record"
    writer = tf.python_io.TFRecordWriter(test_output)
    for example in preparer.get_all_test_entries():
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    
    print(f'Successfully created the TFRecord file: {test_output}')

if __name__ == '__main__':
    tf.app.run()
