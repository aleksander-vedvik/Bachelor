
import os
from prepare_data import Prepare
from PIL import Image


def create_txt_file(entry, output_annotations, output_images):
    filename = entry["filename"]
    image_format = 'JPEG'
    
    image = Image.open(os.path.join(entry["images_path"], '{}'.format(entry["filename"])))
    width, height = image.size
    img = image.convert("RGB")
    img.save(str(output_images + filename), image_format)

    label_name = filename.replace(".jpg", "") + ".txt"

    with open(output_annotations + label_name, "w") as label_file:
        for obj in entry["objects"]:
            class_id = obj['class_id']
            x1, y1 = float(obj['x1']), float(obj['y1'])
            x2, y2 = float(obj['x2']), float(obj['y2'])

            bbox_width = (x2 - x1)
            bbox_height = (y2 - y1)

            center_point_x_normalized = (x1 + bbox_width/2) / width
            center_point_y_normalized = (y1 + bbox_height/2) / height
            
            bbox_width_normalized = bbox_width / width
            bbox_height_normalized = bbox_height / height

            label_file.write(
                f"{class_id} {center_point_x_normalized} {center_point_y_normalized} {bbox_width_normalized} {bbox_height_normalized}\n"
            )
    

def main():
    image_dir = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\Superannotate\\Test\\Test\\images\\'
    anno_path = r'C:\\Users\\Aleks\\Documents\\Bachelor\\Datasets\\Superannotate\\Test\\Test\\annotations.json'
    
    datasets = [ {"dataset": "self_annotated", "images": image_dir, "annotations": anno_path}]

    train_test_distribution = 0.9
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}
    
    preparer = Prepare(train_test_distribution, datasets, classes)
    preparer.stats_dataset("self_annotated")
    
    output_path = "./YOLO/vehicles/"

    for entry in preparer.get_all_train_entries():
        output_annotations = output_path + "labels/train/"
        output_images = output_path + "images/train/"
        
        create_txt_file(entry, output_annotations, output_images)
    for entry in preparer.get_all_test_entries():
        output_annotations = output_path + "labels/val/"
        output_images = output_path + "images/val/"

        create_txt_file(entry, output_annotations, output_images)
    
    print(f'Successfully created the txt files: {output_path}')


if __name__ == '__main__':
    main()
   