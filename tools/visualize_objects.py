import cv2 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import tkinter
matplotlib.use('TkAgg')


def draw_circle(image, object, img_ratio, color=0):
    center_coordinates = (int((float(object["x1"]) + (float(object["x2"]) - float(object["x1"])) / 2)*img_ratio), int((float(object["y1"]) + (float(object["y2"]) - float(object["y1"])) / 2)*img_ratio))
    radius = 0
    if color == 0:
        color_circle = (0, 0, 255)
    else:
        color_circle = color
    thickness_circle = 10

    cv2.circle(image, center_coordinates, radius, color_circle, thickness_circle)


def draw_line(image, start_point, end_point):
    color, thickness = (255,255,255), 2
    cv2.arrowedLine(image, start_point, end_point, color, thickness)


def draw_text(image, track, text):
    object = track.to_tlbr()
    #class_name = track.get_class()

    coordinates = (int(object[0]), int(object[1]-10))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color_text = (255, 255, 255)
    thickness = 2

    cv2.putText(image, text, coordinates, font, fontScale, color_text, thickness)
    """
    text = "ID: " + str(id) + ", " + str(speed) + " km/h"
    coordinates = (int(float(object["x2"])*img_ratio), int(float(object["y2"])*img_ratio))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color_text = (255, 255, 255)
    thickness = 2

    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
    """

def draw_rectangle(image, track, color):
    object = track.to_tlbr()
    
    start_point, end_point = (int(object[0]), int(object[1])), (int(object[2]), int(object[3]))
    color_rectangle = color
    thickness_rectangle = 2
    
    #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    cv2.rectangle(image, start_point, end_point, color_rectangle, thickness_rectangle)
    
    """
    start_point = (int(float(object["x1"])*img_ratio), int(float(object["y1"])*img_ratio))
    end_point = (int(float(object["x2"])*img_ratio), int(float(object["y2"])*img_ratio))
    color_rectangle = (255, 0, 255)
    thickness_rectangle = 1
    
    cv2.rectangle(image, start_point, end_point, color_rectangle, thickness_rectangle)
    """

"""
def visualize_objects(image_np, object_detections, label_offset, w, h, tracker):
    tracker.reset_current()
    for i, score in enumerate(object_detections["detection_scores"]):
        speed = "NA"
        if score >= 0.9:
            x1 = float(object_detections['detection_boxes'][i][1]) * float(w)
            y1 = float(object_detections['detection_boxes'][i][0]) * float(h)
            x2 = float(object_detections['detection_boxes'][i][3]) * float(w)
            y2 = float(object_detections['detection_boxes'][i][2]) * float(h)
            detected_object = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            
            vehicle_id, speed = tracker.track(detected_object)

            draw_text(image_np, detected_object, 1, vehicle_id, speed)
            draw_rectangle(image_np, detected_object, 1)
            draw_circle(image_np, detected_object, 1)

            
    for center_point in tracker.objects_detected:
        color = (0, 255, 0)
        draw_circle(image_np, center_point["coordinates"], 1, color)
    
    tracker.update()
"""