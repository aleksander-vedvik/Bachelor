import cv2
import numpy as np
from prepare_data import Prepare
import os

def draw_rectangle(image, object, img_ratio):
    start_point = (int(float(object["x1"])*img_ratio), int(float(object["y1"])*img_ratio))
    end_point = (int(float(object["x2"])*img_ratio), int(float(object["y2"])*img_ratio))
    color_rectangle = (255, 0, 255)
    thickness_rectangle = 1
    
    cv2.rectangle(image, start_point, end_point, color_rectangle, thickness_rectangle)

def draw_text(image, detected_object, img_ratio):
    text = "Class: " + str(detected_object["class"]) + ", ID: " + detected_object["info"]["ID"]
    coordinates = (int(float(detected_object["x2"])*img_ratio), int(float(detected_object["y2"])*img_ratio))
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
        detected_object = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": obj["class"], "info": obj["info"]}
        
        draw_rectangle(image, detected_object, 1)
        draw_text(image, detected_object, 1)

def gamma(image, gamma=2.0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def main():
    image_dir_other = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\DATA MASTER\\cveet-data\\dark\\'
    anno_path_other = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\DATA MASTER\\cveet-data\\dark\\'

    image_dir_night = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\DATA MASTER\\cveet-data\\raw_night\\'
    anno_path_night = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\DATA MASTER\\cveet-data\\raw_night\\'
    
    image_dir_raw = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\DATA MASTER\\cveet-data\\raw\\'
    anno_path_raw = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\DATA MASTER\\cveet-data\\raw\\'
    
    image_dir_kitti = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\DATA MASTER\\cveet-data\\kitti\\'
    anno_path_kitti = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\DATA MASTER\\cveet-data\\kitti\\'

    image_dir = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\Superannotate\\Test\\Test\\images\\'
    anno_path = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\Superannotate\\Test\\Test\\annotations.json'
    
    image_dir1 = r'F:\\Bachelor\\DATA\\Incidents\\Video11\\images\\'
    anno_path1 = r'F:\\Bachelor\\DATA\\Incidents\\Video11\\annotations.json'

    datasets = [ ]
    
    org_path = r'F:\\Bachelor\\DATA\\Incidents\\Video'
    for i in range(1, 15):
        image_dir1 = org_path + str(i) + "\\images\\"
        anno_path1 = org_path + str(i) + "\\annotations.json"
        dataset_name = "self_annotated" + str(i)
        datasets.append({"dataset": dataset_name, "images": image_dir1, "annotations": anno_path1})
    
    train_test_distribution = 1
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}
    
    preparer = Prepare(train_test_distribution, datasets, classes)
    for ds in datasets:
        preparer.stats_dataset(ds["dataset"])
    
    #file = "5 fps0927.jpg"
    #file = "5 fps0221.jpg"
    file= "5 fps1284.jpg"
    frame = 0
    for example in preparer.get_all_train_entries():
        frame += 1
        if frame < 100:
            continue
        #if file != example["filename"]:
        #    continue
        path = example["images_path"]
        image_path = os.path.join(path, '{}'.format(example["filename"]))
        image = cv2.imread(image_path)

        #variance_list=[15, 80, 30]
        #variance_list=[200, 200, 200]
        variance_list=[1000, 1000, 1000]
        #variance=300
        variance=500

        mask_path = "./output/3/mask.png"
        mask = cv2.imread(mask_path, 0)

        #msr=MSR(image,variance_list)
        #ssr=SSR(image, variance)
        #ssr=retinex_SSR(image, variance)
        #msr = retinex_MSR(image, variance_list)
        #msr = retinex_MSRCR(image, variance_list)
        #msr = retinex_MSRCR(image)
        #msr = retinex_MSRCP(image, variance_list)
        #msr = retinex_MSRCP(image)
        #msr = retinex_AMSR(image, variance_list)
        #gimp = retinex_gimp(image)
        #fm = retinex_FM(image)
        #msr = retinex_FM(image)
        gl = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        he = cv2.equalizeHist(gl)
        nl2 = gamma(image, 2.0)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        #nl1_5 = gamma(image, 1.5)
        
        #visualize_objects(image, example)
        #visualize_objects(img_msr, example)
        
        #visualize_objects(image, example)
        #visualize_objects(src, example)
        #visualize_objects(dst, example)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        #cv2.imshow('MSR', msr)
        #cv2.imshow('FM', fm)
        #cv2.imshow('GIMP', gimp)
        #cv2.imshow('SSR', ssr)
        cv2.imshow('Source image', image)
        #cv2.imshow('Grayscale image', src)
        #cv2.imshow('Histogram Equalized image', dst)

        """cv2.imwrite("./output/2/org.jpg", image)
        cv2.imwrite("./output/2/ssr.jpg", ssr)
        cv2.imwrite("./output/2/msr.jpg", msr)
        cv2.imwrite("./output/2/gl.jpg", gl)
        cv2.imwrite("./output/2/he.jpg", he)
        cv2.imwrite("./output/2/nl2_gray.jpg", nl2)"""
        #cv2.imwrite("./output/3/masked.jpg", masked_image)

        #cv2.imwrite("./output/2/nl1_5.jpg", nl1_5)
        #cv2.imwrite("./output/2/nl2.jpg", nl2)
        #cv2.imwrite("./output/2/ssr.jpg", ssr)
        #cv2.imwrite("./output/src.jpg", image)
        #cv2.imwrite("./output/msr.jpg", msr)
        
    #for example in preparer.get_all_test_entries():
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()