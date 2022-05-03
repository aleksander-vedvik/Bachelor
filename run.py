import os
import time
from absl.flags import FLAGS
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detection_model import Detection_Model
from tracking_model import Tracking_Model
from incident_evaluater import Evaluate_Incidents
from performance_evaluater import Evaluate_Performance
import argparse
from visualize_objects import draw_rectangle, draw_text, draw_line


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
                    help="Choose whether to use a pre-trained model or not. 1 = True, 0 = False (0 is default)",
                    type=str)
                
parser.add_argument("-s",
                    "--skip_frames",
                    help="Choose how many frames should be skipped",
                    type=int)

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
    #image_dir1 = r'F:\\Bachelor\\DATA\\Incidents\\Video2\\images\\'
    #anno_path1 = r'F:\\Bachelor\\DATA\\Incidents\\Video2\\annotations.json'
    #datasets = [{"dataset": "self_annotated1", "images": image_dir1, "annotations": anno_path1}]

    datasets = []
    org_path = r'F:\\Bachelor\\DATA\\Incidents\\Video'
    exclude = [4, 8]
    for i in range(1, 15):
        if i in exclude:
            continue
        image_dir1 = org_path + str(i) + "\\images\\"
        anno_path1 = org_path + str(i) + "\\annotations.json"
        dataset_name = "self_annotated" + str(i)
        datasets.append({"dataset": dataset_name, "images": image_dir1, "annotations": anno_path1})
    
    video_path = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\Video footage\\Test\\Tunnel_opening_night_time_long.mp4'
    #video_path = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\Video footage\\Incidents_4.mp4'
    path_to_this_file = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(path_to_this_file, 'model_data/mars-small128.pb')
    #model_filename = os.path.join(path_to_this_file, 'model_data/model640.pt')
    
    paths = {
        "CHECKPOINT_PATH": "../../Detection/TensorFlowAPI/models/ssd_mobnet/",
        "PIPELINE_CONFIG": "../../Detection/TensorFlowAPI/models/ssd_mobnet/pipeline.config", 
        "LABELMAP": "../../Detection/TensorFlowAPI/annotations/label_map.pbtxt",
        "VIDEO_PATH": video_path,
        "DEEPSORT_MODEL": model_filename
    }
    
    image_enhancement_methods = ["gray_linear", "gray_nonlinear", "he", "retinex_ssr", "retinex_msr", "bgs", "mask"]
    models = ["ssd_mobnet", "ssd_resnet", "faster_rcnn", "mask_rcnn", "yolov5", "efficientdet"]
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}

    video = paths["VIDEO_PATH"]

    model_name = "yolov5"
    tracking_model_name = "DeepSort"

    if args.model in models:
        paths["CHECKPOINT_PATH"] = "../../Detection/TensorFlowAPI/models/" + args.model + "/"
        paths["PIPELINE_CONFIG"] = "../../Detection/TensorFlowAPI/models/" + args.model + "/pipeline.config"
        model_name = args.model

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
        paths["CHECKPOINT_PATH"] = "../../Detection/TensorFlowAPI/pre-trained-models/" + args.model + "/checkpoint/"
        paths["PIPELINE_CONFIG"] = "../../Detection/TensorFlowAPI/pre-trained-models/" + args.model + "/pipeline.config"
        paths["LABELMAP"] = "../../Detection/TensorFlowAPI/annotations/mscoco_label_map.pbtxt"
        model_name = "Pretrained"
        ckpt_number = "0"
        classes = {"car": "3", "truck": "8", "bus": "6", "bike": "2", "person": "1", "motorbike": "4"}
    
    skip_frames = 1
    if args.skip_frames:
        skip_frames = int(args.skip_frames)

    model = Detection_Model(model_name, classes, paths, ckpt_number)
    tracker_model = Tracking_Model(paths["DEEPSORT_MODEL"], tracker_type=tracking_model_name)
    evaluater = Evaluate_Incidents(classes)

    frame_number = 0

    #bgs = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #bgs = cv2.createBackgroundSubtractorKNN()
    #mask = cv2.imread("mask.jpg",0)
    
    #path = [{"video": video}]
    #pe = Evaluate_Performance("Video", path, classes, model, tracker_model)
    pe = Evaluate_Performance("Images", datasets, classes, model, tracker_model)

    while True:
        ret, frame, new_video, mask = pe.read()
        #print(f"Frame: {frame_number}")
        frame_number +=1
        if frame_number % skip_frames != 0:
            continue
        #bgs_mask = bgs.apply(frame)
        
        #bgs_mask = cv2.blur(frame,(3,3))
        
        #frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        #bgs_mask = cv2.bitwise_and(frame, frame, mask=bgs_mask)
        #frame = cv2.bitwise_and(frame, frame, mask=bgs_mask)
        #org_frame = frame

        if ret:
            frame = pe.image_enhancement(frame, image_enhancement, mask)
            """
            if image_enhancement == "gray_linear":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif image_enhancement == "gray_nonlinear":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif image_enhancement == "he":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.equalizeHist(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif image_enhancement == "retinex_ssr":
                variance=200
                img_ssr=SSR(frame, variance)
                frame = cv2.cvtColor(img_ssr, cv2.COLOR_BGR2RGB)
            elif image_enhancement == "retinex_msr":
                variance_list=[100, 100, 100]
                img_msr=MSR(frame, variance_list)
                frame = cv2.cvtColor(img_msr, cv2.COLOR_BGR2RGB)
            elif image_enhancement == "bgs":
                bgs = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
                #bgs = cv2.createBackgroundSubtractorKNN()
                bgs_mask = bgs.apply(frame)
                frame = cv2.bitwise_and(frame, frame, mask=bgs_mask)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif image_enhancement == "mask":
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            """
        else:
            print('Video has ended!')
            break
        #start_time = time.time()

        if new_video:
            print("NEW VIDEO")
            #raise ValueError
            new_tracking_model = Tracking_Model(paths["DEEPSORT_MODEL"], tracker_type=tracking_model_name)
            pe.tracking_model = new_tracking_model

        pe.detect_and_track(frame)

        evaluater.purge(frame_number)

        #for track in tracker_model.get_tracks():
        for track in pe.get_tracks():
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            
            color, text, current_point, next_point = evaluater.evaluate(track, frame_number)

            pe.performance(track, text)
            
            draw_rectangle(frame, track, color)
            draw_text(frame, track, text)
            if current_point and next_point:
                draw_line(frame, current_point, next_point)
            
            
        # calculate frames per second of running detections
        pe.status()
        
        
        #fps = 1.0 / (time.time() - start_time)
        #detection_time = (detection_time - start_time) * 1000
        #fps = round(fps, 1)
        #detection_time = int(detection_time)
        #track_time = int(track_time * 1000)
        #print(f"FPS: {fps}")
        #print(f"Detection time: {detection_time} ms")
        #print(f"Tracking time: {track_time} ms\n")
        #print("FPS: %.2f" % fps)
        #print("Detection time: %.2f s" % detection_time)
        #print("Tracking time: %.2f s\n" % track_time)
        
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output Video", result)

        #saved_image = "F:/Bachelor/Pictures/Yolov5_deepsort/" + str(frame_number) + ".jpg"
        #cv2.imwrite(saved_image, result)
        
        #cv2.imshow("Background subtraction", bgs_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()

    summary = pe.summary()
    print(summary)
    if filename != "":
        output_file = "./output/" + filename + ".txt"
        with open(output_file, "w") as file:
            output = f"Image enhancement: {image_enhancement}\n"
            output += f"Detection: {model_name}\n"
            output += f"Tracking: {tracking_model_name}\n"
            output += summary
            file.write(output)

if __name__ == '__main__':
    main()
