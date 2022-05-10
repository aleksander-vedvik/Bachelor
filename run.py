import os
import sys

PATH_TO_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATH_TO_THIS_FILE + '\\tools\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\tools\\deep_sort')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\training\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\training\\tensorflowapi\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\training\\tensorflowapi\\research\\')
sys.path.insert(0, PATH_TO_THIS_FILE + '\\training\\tensorflowapi\\research\\object_detection')

import cv2
import numpy as np
from tools.detection_model import Detection_Model
from tools.tracking_model import Tracking_Model
from tools.incident_evaluator import Evaluate_Incidents
from tools.performance_evaluator import Evaluate_Performance
import argparse
from tools.visualize_objects import draw_rectangle, draw_text, draw_line

parser = argparse.ArgumentParser(
    description="Real-time detection")
parser.add_argument("-m",
                    "--model",
                    help="Choose what model to use. If not provided, SSD will be used as default.",
                    type=str)

parser.add_argument("-c",
                    "--checkpoint",
                    help="Choose what checkpoint number to use. If not provided, 3 will be used as default.",
                    type=str)

parser.add_argument("-p",
                    "--pretrained",
                    help="Choose whether to use a pre-trained model or not. 1 = True, 0 = False (0 is default).",
                    type=str)
                
parser.add_argument("-s",
                    "--skip_frames",
                    help="Choose how many frames should be skipped.",
                    type=int)

parser.add_argument("-r",
                    "--resize",
                    help="Define a scale factor to resize the input image.",
                    type=float)

parser.add_argument("-t",
                    "--tracking",
                    help="Choose what model to use. If not provided, DeepSort will be used as default.",
                    type=str)

parser.add_argument("-f",
                    "--file",
                    help="A file will be saved with the filename provided. Default is no file saved.",
                    type=str)

parser.add_argument("-i",
                    "--img_enh",
                    help="Specify which image enhancement method. Default is none.",
                    type=str)

args = parser.parse_args()


def main():
    datasets = []
    org_path = r'.\\data\\Incidents\\Video'
    for i in range(1, 13):
        image_dir1 = org_path + str(i) + "\\images\\"
        anno_path1 = org_path + str(i) + "\\annotations.json"
        dataset_name = "self_annotated" + str(i)
        datasets.append({"dataset": dataset_name, "images": image_dir1, "annotations": anno_path1})
    
    model_filename = os.path.join(PATH_TO_THIS_FILE, 'tools/model_data/mars-small128.pb')
    
    paths = {
        "CHECKPOINT_PATH": "./training/models/ssd_mobnet/",
        "PIPELINE_CONFIG": "./training/models/ssd_mobnet/pipeline.config", 
        "LABELMAP": "./training/annotations/label_map.pbtxt",
        "DEEPSORT_MODEL": model_filename
    }
    
    image_enhancement_methods = ["gray_linear", "gray_nonlinear", "he", "retinex_ssr", "retinex_msr", "mask"]
    models = ["ssd_mobnet", "faster_rcnn", "yolov5", "yolov5_trained", "efficientdet"]
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}
    
    model_name = "yolov5"
    if args.model in models:
        paths["CHECKPOINT_PATH"] = "./training/models/" + args.model + "/"
        paths["PIPELINE_CONFIG"] = "./training/models/" + args.model + "/pipeline.config"
        model_name = args.model

    tracking_model_name = "DeepSort"
    if args.tracking:
        tracking_model_name = args.tracking

    ckpt_number = "3"
    if args.checkpoint is not None:
        ckpt_number = args.checkpoint
    
    filename = ""
    if args.file is not None:
        filename = args.file
    
    image_enhancement = "None"
    if args.img_enh is not None and args.img_enh in image_enhancement_methods:
        image_enhancement = args.img_enh
    
    if args.pretrained == "1":
        paths["CHECKPOINT_PATH"] = "./training/pre-trained-models/" + args.model + "/checkpoint/"
        paths["PIPELINE_CONFIG"] = "./training/pre-trained-models/" + args.model + "/pipeline.config"
        paths["LABELMAP"] = "./training/annotations/mscoco_label_map.pbtxt"
        model_name = "Pretrained"
        ckpt_number = "0"
        classes = {"car": "3", "truck": "8", "bus": "6", "bike": "2", "person": "1", "motorbike": "4"}
    
    skip_frames = 1
    if args.skip_frames:
        skip_frames = int(args.skip_frames)
    
    resize = 1
    if args.resize:
        resize = float(args.resize)

    model = Detection_Model(model_name, classes, paths, ckpt_number)
    tracker_model = Tracking_Model(paths["DEEPSORT_MODEL"], tracker_type=tracking_model_name)
    evaluater = Evaluate_Incidents(classes)
    pe = Evaluate_Performance("Images", datasets, classes, model, tracker_model)

    frame_number = 0
    while True:
        ret, frame, new_video, mask = pe.read(resize)
        frame_number +=1
        if frame_number % skip_frames != 0:
            continue

        if ret:
            frame = pe.image_enhancement(frame, image_enhancement, mask)
        else:
            print('Video has ended!')
            break

        if new_video:
            new_tracking_model = Tracking_Model(paths["DEEPSORT_MODEL"], tracker_type=tracking_model_name)
            pe.tracking_model = new_tracking_model

        pe.detect_and_track(frame)

        evaluater.purge(frame_number)

        for track in pe.get_tracks():
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            
            color, text, current_point, next_point = evaluater.evaluate(track, frame_number)

            pe.performance(track, text)
            
            draw_rectangle(frame, track, color)
            draw_text(frame, track, text)
            if current_point and next_point:
                draw_line(frame, current_point, next_point)
            
            
        pe.status()
        
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output Video", result)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()

    summary = pe.summary()
    print(summary)
    if filename != "":
        output_file = "./data/output/" + filename + ".txt"
        with open(output_file, "w") as file:
            output = f"Image enhancement: {image_enhancement}\n"
            output += f"Detection: {model_name}\n"
            output += f"Tracking: {tracking_model_name}\n"
            output += summary
            file.write(output)

if __name__ == '__main__':
    main()
