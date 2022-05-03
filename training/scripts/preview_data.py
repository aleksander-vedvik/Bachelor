import cv2
from prepare_data import Prepare
import os

def draw_rectangle(image, object, img_ratio):
    start_point = (int(float(object["x1"])*img_ratio), int(float(object["y1"])*img_ratio))
    end_point = (int(float(object["x2"])*img_ratio), int(float(object["y2"])*img_ratio))
    color_rectangle = (255, 0, 255)
    thickness_rectangle = 1
    
    cv2.rectangle(image, start_point, end_point, color_rectangle, thickness_rectangle)

def draw_text(image, detected_object, img_ratio):
    try:
        text = "Class: " + str(detected_object["class"]) + ", ID: " + detected_object["info"]["ID"]
    except Exception as e:
        print(e)
        text = str(detected_object["class"])
    coordinates = (int(float(detected_object["x1"])*img_ratio), int(float(detected_object["y2"])*img_ratio))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color_text = (255, 255, 255)
    thickness = 2

    cv2.putText(image, text, coordinates, font, fontScale, color_text, thickness)

def visualize_objects(image, example):
    for obj in example["objects"]:
        x1 = float(obj['x1'])
        x2 = float(obj['x2'])
        y1 = float(obj['y1'])
        y2 = float(obj['y2'])
        detected_object = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": obj["class"], "info": obj.get("info")}
        
        draw_rectangle(image, detected_object, 1)
        draw_text(image, detected_object, 1)


def main():
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
    
    train_test_distribution = 1
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}
    
    preparer = Prepare(train_test_distribution, datasets, classes)
    for ds in datasets:
        preparer.stats_dataset(ds["dataset"])
    
    for example in preparer.get_all_train_entries():
        path = example["images_path"]
        image_path = os.path.join(path, '{}'.format(example["filename"]))
        image = cv2.imread(image_path)

        visualize_objects(image, example)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        cv2.imshow('Source image', image)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()