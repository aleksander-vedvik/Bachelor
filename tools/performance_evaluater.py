import math
import json
import cv2
import os
import time
import numpy as np
from tools.retinex import SSR
from tools.retinex import MSR

class Evaluate_Performance:
    def __init__(self, type, dataset_path, classes, detection_model, tracking_model):
        self.vid = None
        self.width = 0
        self.height = 0
        self.scale = 1
        self.entries = []
        self.next_entry_index = 0
        self.type = type
        self.detected_objects = []
        self.detected_objects_previous = {}
        self.dataset_paths = dataset_path
        self.datasets = {}
        self.classes = classes
        self.detection_model = detection_model
        self.tracking_model = tracking_model
        self.current_video = ""
        self.prepare()

        self.detection_time_current = 0
        self.tracking_time_current = 0
        self.total_time_current = 0
        self.fps_current = 0

        self.image_enhancement_current = 0
        self.mean_image_enhancement_time = 0

        self.mean_detection_time = 0
        self.min_detection_time = -1
        self.max_detection_time = 0
        
        self.mean_tracking_time = 0
        self.min_tracking_time = -1
        self.max_tracking_time = 0
        
        self.mean_total_time = 0
        self.min_total_time = -1
        self.max_total_time = 0

        self.missed_detections = 0
        self.total_number_of_real_detections = 0
        self.total_number_of_detections = 0
        self.total_number_of_detections_adjusted = 0
        self.false_positives_detections = 0 
        self.false_positives_detections_previous = 0 

        self.missed_tracks = 0

        self.detection_accuracy = 0  # IoU: Area of each bounding box
        self.detection_accuracy_adjusted = 0  # IoU: Area of each bounding box
        self.tracking_accuracy = 0   # Total number of real IDs / Total number of detected IDs <-- Whichever is lower
        self.tracking_id_switches = 0   # Total number of real IDs / Total number of detected IDs <-- Whichever is lower
        self.tracking_id_duplicates = 0
        self.incident_accuracy = 0
        self.missed_incidents = 0
        self.false_alarms = 0

        self.mean_fps = 0
        self.min_fps = -1
        self.max_fps = 0

        self.number_of_frames = 0

    @property
    def dataset_paths(self):
        return self._dataset_paths

    @dataset_paths.setter
    def dataset_paths(self, datasets):
        dataset_paths = {}
        for dataset in datasets:
            images = dataset.get("images")
            annotations = dataset.get("annotations")
            dataset_name = dataset.get("dataset")

            video = dataset.get("video")
            if video:
                dataset_paths["video"] = video
            elif images is None or annotations is None or dataset_name is None:
                continue
            else:
                dataset_paths[dataset_name] = {"images": images, "annotations": annotations}
        
        self._dataset_paths = dataset_paths

    def prepare(self):
        if self.type == "Video":
            self.vid = cv2.VideoCapture(self.dataset_paths.get("video"))
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self.type == "Images":
            for dataset_name in self.dataset_paths:
                try:
                    if "self_annotated" in dataset_name:
                        self.prepare_self_annotated(dataset_name)
                except Exception as e:
                    print(e)
                    self.datasets[dataset_name] = {"entries": []}
            self.prepare_all_entries()

    def prepare_self_annotated(self, dataset_name="self_annotated"):
        dataset = self.dataset_paths.get(dataset_name)
        if dataset is None:
            return
        anno_path = dataset.get("annotations")
        img_path = dataset.get("images")
        mask_path = anno_path.replace("annotations.json", "mask.png")
        #class_ids = list(self.classes.values())
        #class_names = list(self.classes.keys())

        if anno_path is None:
            return
        
        with open(anno_path, "r") as annotations:
            data = json.load(annotations)

        annotation_classes_path = anno_path.replace("annotations", "classes")
        with open(annotation_classes_path, "r") as annotation_classes:
            annotation_classes = json.load(annotation_classes)
            
        images_list = {"entries": []}
        for i, img in enumerate(data):
            if i <= 0:
                continue
            filename = img
            row = {"images_path": img_path, "filename": filename, "objects": [], "mask_path": mask_path }
        
            for object in data[img]['instances']:

                info = {}
                for class_ in annotation_classes:
                    if class_["id"] == object["classId"]:
                        class_name = class_["name"]

                        for object_attribute in object["attributes"]:
                            for attribute_group in class_["attribute_groups"]:
                                if object_attribute["groupId"] == attribute_group["id"]:
                                    for attribute_ in attribute_group["attributes"]:
                                        if attribute_["id"] == object_attribute["id"]:
                                            info[attribute_group["name"]] = attribute_["name"]
                
                #print(info)
                            #if attribute_group["id"] == attribute["groupId"]:
                                #for attribute in attribute_group["attributes"]:
                                    #if attribute["id"] == 1:  # True
                                        #print(attribute["name"])


                #class_name = class_names[class_ids.index((str(object["classId"])))]
                #class_name = class_name.lower()
                if class_name == "people":
                    class_name = "person"

                if class_name not in self.classes:
                    continue
                    
                x1 = float(object["points"]["x1"])
                y1 = float(object["points"]["y1"])
                x2 = float(object["points"]["x2"])
                y2 = float(object["points"]["y2"])

                row["objects"].append({"class": class_name, "class_id": self.classes.get(class_name),  "x1": x1, "y1": y1, "x2": x2, "y2": y2, "info": info})
            
            images_list["entries"].append(row)

        #sort(images_list)
        images_list['entries'] = sorted(images_list['entries'], key = lambda i: i['filename'])
        self.datasets[dataset_name] = images_list

    def prepare_all_entries(self):
        if self.type != "Images":
            return

        print("\nEntries:")
        classes = {}
        number_of_objects = 0
        entries = []
        for dataset in self.datasets:
            for entry in self.datasets[dataset]["entries"]:
                entries.append(entry)
                number_of_objects += len(entry["objects"])
                for obj in entry["objects"]:
                    if obj["class"] in classes:
                        classes[obj["class"]] += 1
                    else:
                        classes[obj["class"]] = 1
        
        print(f"Number of files: {len(entries)}")
        print(f"Number of objects: {number_of_objects}")
        for obj_class in classes:
            print(f" - {obj_class}: {classes[obj_class]}")
        #self.entries = sorted(entries, key = lambda i: i['filename'])
        self.entries = entries

    def performance(self, track, text):
        bbox = track.to_tlbr()
        object_class = track.get_class()
        track_id = track.track_id
        
        #print("\nDetection")
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        incident = False
        #detected_objects_id = []
        best_IoU = {"score": 0, "object": None, "real_object": None}
        for real_object in self.entries[self.next_entry_index-1]["objects"]:
            real_object["x1"] *= self.scale
            real_object["y1"] *= self.scale
            real_object["x2"] *= self.scale
            real_object["y2"] *= self.scale
            if x1 > real_object["x1"]: 
                x_min = x1
            else:                      
                x_min = real_object["x1"]
            if y1 > real_object["y1"]: 
                y_min = y1
            else:                      
                y_min = real_object["y1"]
            if x2 < real_object["x2"]: 
                x_max = x2
            else:                      
                x_max = real_object["x2"]
            if y2 < real_object["y2"]: 
                y_max = y2
            else:                      
                y_max = real_object["y2"]
            
            intersection_area = (x_max - x_min) * (y_max - y_min)
            if intersection_area < 0 or (x_max - x_min) < 0 or (y_max - y_min) < 0:
                continue
            
            union_area = ((real_object["x2"] - real_object["x1"]) * (real_object["y2"] - real_object["y1"])) + ((x2 - x1) * (y2 - y1)) - intersection_area
            if union_area < 0:
                print(f"IA: {intersection_area}")
                print(f"UA: {union_area}")
                print(f"RO: x1 = {real_object['x1']}, y1 = {real_object['y1']}, x2 = {real_object['x2']}, y2 = {real_object['y2']}")
                print(f"DO: x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")
                raise ValueError

            #detection_accuracy = intersection_area / union_area
            IoU = intersection_area / union_area

            #if real_object['info']['ID'] in detected_objects_id:
            #    print(real_object['info']['ID'])
            #    print(detected_objects_id)
            #    print("DUPLICATE")
            #    raise ValueError
            #detected_objects_id.append(real_object['info']['ID'])
            
            #if best_IoU["object"] is not None and (IoU - 1)**2 < (best_IoU["score"] - 1)**2:
            if (IoU - 1)**2 < (best_IoU["score"] - 1)**2:
                best_IoU["object"] = {"bbox": bbox, "class": object_class, "ID": track_id}
                best_IoU["real_object"] = real_object
                best_IoU["score"] = IoU
            #else:
            #    best_IoU["object"] = {"bbox": bbox, "class": object_class}
            #    best_IoU["real_object"] = real_object
            #    best_IoU["score"] = IoU
            #print(f"Detection accuracy: {detection_accuracy * 100}")
        if best_IoU["object"] is not None and best_IoU["score"] > 0.4:
            """
            print(f"Detection accuracy: {round(best_IoU['score'] * 100, 1)} %")
            #print(f"Real object: {best_IoU['object']}")
            print(f"DO: {best_IoU['object']['class']} // RO: {best_IoU['object']['class']}") 
            print(f"\t- x1 = {round(best_IoU['object']['bbox'][0], 2)} // {round(best_IoU['real_object']['x1'], 2)}")
            print(f"\t- y1 = {round(best_IoU['object']['bbox'][1], 2)} // {round(best_IoU['real_object']['y1'], 2)}")
            print(f"\t- x2 = {round(best_IoU['object']['bbox'][2], 2)} // {round(best_IoU['real_object']['x2'], 2)}")
            print(f"\t- y2 = {round(best_IoU['object']['bbox'][3], 2)} // {round(best_IoU['real_object']['y2'], 2)}")
            #print(f"Detected object class: {object_class}")
            #print(f"Detected object: {detection}")
            """
            if best_IoU["real_object"]['info']['status'] == "Incident":
                incident = True
            self.detected_objects.append(best_IoU)
        else:
            self.false_positives_detections += 1
        
        if incident and ("Stopped vehicle" in text or "Pedestrian" in text):
            self.incident_accuracy += 1
        elif incident:
            self.missed_incidents += 1
        elif "Stopped vehicle" in text or "Pedestrian" in text:
            self.false_alarms += 1

    """
    def IoU(self, model_detections):
        for detection, object_class in zip(model_detections["boxes"], model_detections["object_classes"]):
            print("\nDetection")
            x1 = detection[0]
            y1 = detection[1]
            x2 = x1 + detection[2]
            y2 = y1 + detection[3]

            detected_objects_id = []
            best_IoU = {"score": 0, "object": None}
            for real_object in self.entries[self.next_entry_index-1]["objects"]:
                if x1 > real_object["x1"]: 
                    x_min = x1
                else:                      
                    x_min = real_object["x1"]
                if y1 > real_object["y1"]: 
                    y_min = y1
                else:                      
                    y_min = real_object["y1"]
                if x2 < real_object["x2"]: 
                    x_max = x2
                else:                      
                    x_max = real_object["x2"]
                if y2 < real_object["y2"]: 
                    y_max = y2
                else:                      
                    y_max = real_object["y2"]
                
                intersection_area = (x_max - x_min) * (y_max - y_min)
                if intersection_area < 0 or (x_max - x_min) < 0 or (y_max - y_min) < 0:
                    continue
                
                union_area = ((real_object["x2"] - real_object["x1"]) * (real_object["y2"] - real_object["y1"])) + ((x2 - x1) * (y2 - y1)) - intersection_area
                if union_area < 0:
                    print(f"IA: {intersection_area}")
                    print(f"UA: {union_area}")
                    print(f"RO: x1 = {real_object['x1']}, y1 = {real_object['y1']}, x2 = {real_object['x2']}, y2 = {real_object['y2']}")
                    print(f"DO: x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")
                    raise ValueError

                #detection_accuracy = intersection_area / union_area
                IoU = intersection_area / union_area

                if real_object['info']['ID'] in detected_objects_id:
                    print(real_object['info']['ID'])
                    print(detected_objects_id)
                    print("DUPLICATE")
                    raise ValueError
                detected_objects_id.append(real_object['info']['ID'])
                
                if best_IoU["object"] is not None and (IoU - 1)**2 < (best_IoU["score"] - 1)**2:
                    best_IoU["object"] = real_object
                    best_IoU["score"] = IoU
                else:
                    best_IoU["object"] = real_object
                    best_IoU["score"] = IoU
                #print(f"Detection accuracy: {detection_accuracy * 100}")
            if best_IoU["object"] is not None and best_IoU["score"] > 0.5:
                print(f"Detection accuracy: {round(best_IoU['score'] * 100, 1)} %")
                #print(f"Real object: {best_IoU['object']}")
                print(f"DO: {object_class} // RO: {best_IoU['object']['class']}") 
                print(f"\t- x1 = {round(x1, 2)} // {round(best_IoU['object']['x1'], 2)}")
                print(f"\t- y1 = {round(y1, 2)} // {round(best_IoU['object']['y1'], 2)}")
                print(f"\t- x2 = {round(x2, 2)} // {round(best_IoU['object']['x2'], 2)}")
                print(f"\t- y2 = {round(y2, 2)} // {round(best_IoU['object']['y2'], 2)}")
                #print(f"Detected object class: {object_class}")
                #print(f"Detected object: {detection}")
            else:
                self.false_positives_detections += 1
    """

    def image_enhancement(self, frame, image_enhancement="", mask=None):
        img_enh_start = time.time()
        if image_enhancement == "gray_linear":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif image_enhancement == "gray_nonlinear":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gamma=2.0
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif image_enhancement == "he":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.equalizeHist(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif image_enhancement == "retinex_ssr":
            #self.scale = 0.5
            #self.width = int(self.width * self.scale)
            #self.height = int(self.height * self.scale)
            #dim = (self.width, self.height)
            #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            variance=300
            img_ssr=SSR(frame, variance)
            frame = cv2.cvtColor(img_ssr, cv2.COLOR_BGR2RGB)
        elif image_enhancement == "retinex_msr":
            #self.scale = 0.5
            #width = int(self.width * self.scale)
            #height = int(self.height * self.scale)
            #dim = (width, height)
            #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            variance_list=[200, 200, 200]
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
        
        img_enh_end = time.time()
        self.image_enhancement_current = img_enh_end - img_enh_start
        self.mean_image_enhancement_time += self.image_enhancement_current

        return frame

    def detect(self, frame):
        detection_start = time.time()
        model_detections = self.detection_model.detect(frame, self.width, self.height)
        detection_end = time.time()
        self.detection_time_current = detection_end - detection_start

        if self.detection_time_current < 10:
            self.mean_detection_time += self.detection_time_current

        if self.min_detection_time == -1 or (self.detection_time_current < self.min_detection_time and self.detection_time_current > 0):
            self.min_detection_time = self.detection_time_current 
        if self.detection_time_current > self.max_detection_time and self.detection_time_current < 10:
            self.max_detection_time = self.detection_time_current

        return model_detections
    
    def track(self, model_detections):
        track_start = time.time()
        self.tracking_model.track(model_detections)
        track_end = time.time()
        self.tracking_time_current = track_end - track_start

        self.mean_tracking_time += self.tracking_time_current

        if (self.min_tracking_time == -1 or self.tracking_time_current < self.min_tracking_time) and self.tracking_time_current > 0:
            self.min_tracking_time = self.tracking_time_current 
        if self.tracking_time_current > self.max_tracking_time:
            self.max_tracking_time = self.tracking_time_current

    def detect_and_track(self, frame):
        self.number_of_frames += 1
        model_detections = self.detect(frame)
        self.track(model_detections)

        self.total_time_current = self.detection_time_current + self.tracking_time_current + self.image_enhancement_current
        self.mean_total_time += self.total_time_current

        if (self.min_total_time == -1 or self.total_time_current < self.min_total_time) and self.total_time_current > 0:
            self.min_total_time = self.total_time_current
        if self.total_time_current > self.max_total_time:
            self.max_total_time = self.total_time_current

        self.fps_current = 1.0 / (self.total_time_current)
        self.fps_current = round(self.fps_current, 3)

        self.mean_fps += self.fps_current
        if (self.min_fps == -1 or self.fps_current < self.min_fps) and self.fps_current > 0:
            self.min_fps = self.fps_current
        if self.fps_current > self.max_fps:
            self.max_fps = self.fps_current
        
        #self.IoU()

    def read(self):
        if self.type == "Video":
            return True, self.vid.read(), False
        else:
            #time.sleep(1)
            frame = None
            ret = False
            new_video = False
            mask = None
            try:
                entry = self.entries[self.next_entry_index]
                path = entry["images_path"]
                mask_path = entry["mask_path"]
                image_path = os.path.join(path, '{}'.format(entry["filename"]))
                frame = cv2.imread(image_path)
                self.height, self.width, _ = frame.shape
                mask = cv2.imread(mask_path, 0)
                ret = True

                if self.current_video != entry['images_path'] and self.current_video != "":
                    new_video = True
                self.current_video = entry['images_path']
                #frame = cv2.VideoCapture(image_path)
            except IndexError as e:
                print(e)
            self.next_entry_index += 1
            return ret, frame, new_video, mask
    
    def get_tracks(self):
        return self.tracking_model.get_tracks()
    
    def status(self):
        detection_time = int((self.detection_time_current) * 1000)
        track_time = int(self.tracking_time_current * 1000)
        print(f"\nFrame: {self.number_of_frames}")
        print(f"FPS: {self.fps_current}")
        print(f"IE time: {int(self.image_enhancement_current* 1000)} ms")
        print(f"Detection time: {detection_time} ms")
        print(f"Tracking time: {track_time} ms")
        print(f"Total time: {int(self.total_time_current* 1000)} ms")

        avg_score = 0
        avg_score_adjusted = 0
        number_of_detections_adjusted = 0
        number_of_correct_classes = 0
        number_of_wrong_classes = 0
        number_of_correct_ids = 0
        number_of_wrong_ids = 0
        number_of_duplicate_ids = 0
        object_ids = []
        print("Detected objects:")
        for detected_object in self.detected_objects:
            print(f"\t- {detected_object['object']['class']}, {round(detected_object['score']*100, 2)} %")
            avg_score += detected_object['score']
            if detected_object["real_object"]['info']['occluded'] == "False":
                avg_score_adjusted += detected_object['score']
                number_of_detections_adjusted += 1
            if detected_object["object"]["class"] == detected_object["real_object"]["class"]:
                print("\t\t- Correct Class")
                number_of_correct_classes += 1
            else:
                print("\t\t- Wrong Class")
                number_of_wrong_classes += 1
            
            #if detected_object['object']['ID'] in self.detected_objects_previous:
            if detected_object['real_object']['info']['ID'] in self.detected_objects_previous:
                #if self.detected_objects_previous[detected_object['object']['ID']] == detected_object['real_object']['info']['ID']:
                if self.detected_objects_previous[detected_object['real_object']['info']['ID']] == detected_object['object']['ID']:
                    if detected_object['real_object']['info']['ID'] in object_ids:
                        print("\t\t- Duplicate ID")
                        number_of_duplicate_ids += 1
                    else:
                        print("\t\t- Correct ID")
                        number_of_correct_ids += 1
                else:
                    print("\t\t- Wrong ID")
                    if detected_object["real_object"]["class"] != "person":  # It is not necessary to track ID of people and thus the result should be unaffected.
                        number_of_wrong_ids += 1
                    self.detected_objects_previous[detected_object['real_object']['info']['ID']] = detected_object['object']['ID']
                    #self.detected_objects_previous[detected_object['object']['ID']] = detected_object['real_object']['info']['ID']
            else:
                self.detected_objects_previous[detected_object['real_object']['info']['ID']] = detected_object['object']['ID']
                #self.detected_objects_previous[detected_object['object']['ID']] = detected_object['real_object']['info']['ID']
            object_ids.append(detected_object['real_object']['info']['ID'])

            """
            for previous_detection in self.detected_objects_previous:
                if detected_object['object']['ID'] == previous_detection['object']['ID']:
                    if detected_object['real_object']['info']['ID'] == previous_detection['real_object']['info']['ID']:
                        print("\t\t- Correct ID")
                        number_of_correct_ids += 1
                    else:
                        print("\t\t- Wrong ID")
                        number_of_wrong_ids += 1
                    break
            """
        self.tracking_accuracy += number_of_correct_ids
        self.tracking_id_switches += number_of_wrong_ids
        self.tracking_id_duplicates += number_of_duplicate_ids

        self.detection_accuracy += avg_score
        self.detection_accuracy_adjusted += avg_score_adjusted
        self.total_number_of_detections += len(self.detected_objects)
        self.total_number_of_detections_adjusted += number_of_detections_adjusted
        if len(self.detected_objects): avg_score /= len(self.detected_objects)
        if number_of_detections_adjusted > 0: avg_score_adjusted /= number_of_detections_adjusted
        print(f"Average score: {round(avg_score*100, 2)} %")
        print(f"Average score adjusted: {round(avg_score_adjusted*100, 2)} %")

        tmp_missed = 0
        for real_object in self.entries[self.next_entry_index-1]["objects"]:
            if real_object['info']['ID'] not in object_ids:
                self.missed_detections += 1
                tmp_missed += 1
                #if real_object['info']['status'] == "Incident":
                #    self.missed_incidents += 1
        self.total_number_of_real_detections += len(self.entries[self.next_entry_index-1]["objects"])

        print(f"Missed detections: {tmp_missed}")
        try:
            print(f"Missed detections: {round(100*tmp_missed/len(self.entries[self.next_entry_index-1]['objects']), 1)} %")
        except Exception as e:
            print(e)
        print(f"False positive detections: {self.false_positives_detections - self.false_positives_detections_previous}")

        #self.detected_objects_previous = self.detected_objects.copy()
        self.detected_objects = []

        self.false_positives_detections_previous = self.false_positives_detections
        
    def summary(self):
        text = "\n"
        try:
            text += f"Scale: {int(self.scale*100)} %\n"
            text += f"Resolution: {int(self.height*self.scale)}x{int(self.width*self.scale)} px\n"
            text += f"Mean image enhancement time: {int(1000 * self.mean_image_enhancement_time / self.number_of_frames)} ms\n"
            text += "\n"
            text += f"Mean detection time: \t{int(1000 * self.mean_detection_time / self.number_of_frames)} ms\n"
            text += f"Min detection time: \t{int(1000 * self.min_detection_time)} ms\n"
            text += f"Max detection time: \t{int(1000 * self.max_detection_time)} ms\n"
            text += "\n"
            text += f"Mean tracking time: \t{int(1000 * self.mean_tracking_time / self.number_of_frames)} ms\n"
            text += f"Min tracking time: \t\t{int(1000 * self.min_tracking_time)} ms\n"
            text += f"Max tracking time: \t\t{int(1000 * self.max_tracking_time)} ms\n"
            text += "\n"
            text += f"Mean total time: \t{int(1000 * self.mean_total_time / self.number_of_frames)} ms\n"
            text += f"Min total time: \t{int(1000 * self.min_total_time)} ms\n"
            text += f"Max total time: \t{int(1000 * self.max_total_time)} ms\n"
            text += "\n"
            text += f"Mean fps: \t{round(self.mean_fps / self.number_of_frames, 1)}\n"
            text += f"Min fps: \t{int(self.min_fps)}\n"
            text += f"Max fps: \t{int(self.max_fps)}\n"
            text += "\n"
            text += f"False positive detections: \t{round(100*self.false_positives_detections/self.total_number_of_detections, 1)} %\n"
            text += f"Missed detections: \t\t\t{round(100*self.missed_detections/self.total_number_of_real_detections, 1)} %\n"
            text += "\n"
            text += f"Detection accuracy: \t\t\t{round(100*self.detection_accuracy/self.total_number_of_detections, 1)} %\n"
            text += f"Detection accuracy adjusted: \t{round(100*self.detection_accuracy_adjusted/self.total_number_of_detections_adjusted, 1)} %\n"
            text += f"Tracking accuracy: \t\t\t\t{round(100*self.tracking_accuracy/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %\n"
            text += f"Tracking ID duplicates: \t\t{round(100*self.tracking_id_duplicates/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %\n"
            text += f"Tracking ID switches: \t\t\t{round(100*self.tracking_id_switches/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %\n"
            text += "\n"
            text += f"Incident accuracy: \t{round(100*self.incident_accuracy/(self.incident_accuracy+self.missed_incidents), 1)} %\n"
            text += f"Missed incidents: \t{round(100*self.missed_incidents/(self.incident_accuracy+self.missed_incidents), 1)} %\n"
            text += f"False alarms: \t\t{round(100*self.false_alarms/self.total_number_of_detections, 1)} %\n"
            text += "\n"
            text += f"Total number of detections: {self.total_number_of_detections}\n"
            
            """
            print("\n")
            print(f"Mean detection time: \t{int(1000 * self.mean_detection_time / self.number_of_frames)} ms")
            print(f"Min detection time: \t{int(1000 * self.min_detection_time)} ms")
            print(f"Max detection time: \t{int(1000 * self.max_detection_time)} ms")
            print("")
            print(f"Mean tracking time: \t{int(1000 * self.mean_tracking_time / self.number_of_frames)} ms")
            print(f"Min tracking time: \t{int(1000 * self.min_tracking_time)} ms")
            print(f"Max tracking time: \t{int(1000 * self.max_tracking_time)} ms")
            print("")
            print(f"Mean total time: \t{int(1000 * self.mean_total_time / self.number_of_frames)} ms")
            print(f"Min total time: \t{int(1000 * self.min_total_time)} ms")
            print(f"Max total time: \t{int(1000 * self.max_total_time)} ms")
            print("")
            print(f"Mean fps: \t{int(self.mean_fps / self.number_of_frames)}")
            print(f"Min fps: \t{int(self.min_fps)}")
            print(f"Max fps: \t{int(self.max_fps)}")
            print("")
            print(f"False positive detections: \t{self.false_positives_detections}")
            print(f"Missed detections: \t\t{round(100*self.missed_detections/self.total_number_of_real_detections, 1)} %")
            print("")
            print(f"Detection accuracy: \t\t{round(100*self.detection_accuracy/self.total_number_of_detections, 1)} %")
            print(f"Detection accuracy adjusted: \t{round(100*self.detection_accuracy_adjusted/self.total_number_of_detections_adjusted, 1)} %")
            print(f"Tracking accuracy: \t\t{round(100*self.tracking_accuracy/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %")
            print(f"Tracking ID duplicates: \t{round(100*self.tracking_id_duplicates/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %")
            print(f"Tracking ID switches: \t\t{round(100*self.tracking_id_switches/(self.tracking_accuracy+self.tracking_id_switches+self.tracking_id_duplicates), 1)} %")
            print("")
            print(f"Incident accuracy: \t{round(100*self.incident_accuracy/(self.incident_accuracy+self.missed_incidents), 1)} %")
            print(f"Missed incidents: \t{round(100*self.missed_incidents/(self.incident_accuracy+self.missed_incidents), 1)} %")
            print(f"False alarms: \t\t{self.false_alarms}")
            """
        except Exception as e:
            print(e)

        return text
        #self.incident_accuracy = 0
        #self.false_alarms = 0


#for class_ in ANNOTATION_CLASSES:
#    if class_["id"] == 1:  # CAR
#        for attribute_group in class_["attribute_groups"]:
#            if attribute_group["id"] == 1:  # Occluded
#                for attribute in attribute_group["attributes"]:
#                    if attribute["id"] == 1:  # True
#                        print(attribute["name"])
        
#        object["classId"] == 1 <-- CAR
#        for attribute in object["attributes"]:
#            if attribute["groupId"] == attribute_group["id"]:
#                if attribute["id"] == attribute["id"]:
#                    print(True)
#    if attribute_group["id"] == 2: PERSON   
#    if attribute_group["id"] == 3: TRUCK
#    if attribute_group["id"] == 4: BUS   
#    if attribute_group["id"] == 5: BIKE   
#    if attribute_group["id"] == 6: MOTORBIKE   


