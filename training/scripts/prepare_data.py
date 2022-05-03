import json
import glob
import xml.etree.ElementTree as ET


class Prepare:
    def __init__(self, train_test_distribution=float, dataset_paths=list, classes=dict):
        self.train_test_distribution = train_test_distribution
        self.dataset_paths = dataset_paths
        self.datasets = {}
        self.classes = classes
        self.prepare_self_annotated()
        self.prepare_rest()

    @property
    def train_test_distribution(self):
        return self._train_test_distribution

    @train_test_distribution.setter
    def train_test_distribution(self, train_test_distribution=0.9):
        tt_dist = train_test_distribution
        if tt_dist is None or 1 < tt_dist < 0:
            tt_dist = 0.9
        self._train_test_distribution = tt_dist

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
            if images is None or annotations is None or dataset_name is None:
                continue
            else:
                dataset_paths[dataset_name] = {"images": images, "annotations": annotations}
        
        self._dataset_paths = dataset_paths

    def prepare_rest(self):
        for dataset_name in self.dataset_paths:
            if dataset_name == "self_annotated":
                continue
            try:
                if "self_annotated" in dataset_name:
                    self.prepare_self_annotated(dataset_name)
                else:
                    self.prepare_other(dataset_name)
            except Exception as e:
                print(e)
                self.datasets[dataset_name] = {"train": [], "test": []}

    def prepare_other(self, other_name="other"):
        dataset = self.dataset_paths.get(other_name)
        if dataset is None:
            return
        path = dataset.get("annotations")
        img_path = dataset.get("images")
        train_test_distribution = self.train_test_distribution
        if path is None:
            return
            
        images_list = {"train": [], "test": []}
        images_tmp = []
        for n, file in enumerate(glob.glob(path + '/*.xml')):
            tree = ET.parse(file)
            root = tree.getroot()
            
            filename = root[1].text

            row = {"images_path": img_path, "filename": filename, "objects": [] }
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_name = class_name.lower()
                if class_name == "people":
                    class_name = "person"
                if class_name not in self.classes:
                    continue
                x1 = float(obj.find('bndbox').find('xmin').text)
                y1 = float(obj.find('bndbox').find('ymin').text)
                x2 = float(obj.find('bndbox').find('xmax').text)
                y2 = float(obj.find('bndbox').find('ymax').text)

                if x2 < x1 or y2 < y1:
                    print("\nBAD! x2 must be higher than x1, and y2 must be higher than y1!")
                    print(f"Filename: {filename}")
                    print(x1, x2, y1, y2)
                    

                row["objects"].append({"class": class_name, "class_id": self.classes.get(class_name), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                images_tmp.append(row)
            
        number_of_files = len(images_tmp)
        for i, image_info in enumerate(images_tmp):
            if i / number_of_files >= train_test_distribution:
                images_list["test"].append(image_info)
            else:
                images_list["train"].append(image_info)

        self.datasets[other_name] = images_list

    def prepare_self_annotated(self, dataset_name="self_annotated"):
        dataset = self.dataset_paths.get(dataset_name)
        if dataset is None:
            return
        anno_path = dataset.get("annotations")
        img_path = dataset.get("images")
        train_test_distribution = self.train_test_distribution

        if anno_path is None:
            return
        
        with open(anno_path, "r") as annotations:
            data = json.load(annotations)
        
        annotation_classes_path = anno_path.replace("annotations", "classes")
        with open(annotation_classes_path, "r") as annotation_classes:
            annotation_classes = json.load(annotation_classes)
            
        images_list = {"train": [], "test": []}
        for i, img in enumerate(data):
            if i <= 0:
                continue
            filename = img
            row = {"images_path": img_path, "filename": filename, "objects": [] }
        
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

                if class_name == "people":
                    class_name = "person"

                if class_name not in self.classes:
                    continue
                    
                x1 = float(object["points"]["x1"])
                y1 = float(object["points"]["y1"])
                x2 = float(object["points"]["x2"])
                y2 = float(object["points"]["y2"])

                row["objects"].append({"class": class_name, "class_id": self.classes.get(class_name),  "x1": x1, "y1": y1, "x2": x2, "y2": y2, "info": info})
            
            if i / len(data) >= train_test_distribution:
                images_list["test"].append(row)
            else:
                images_list["train"].append(row)

        images_list['train'] = sorted(images_list['train'], key = lambda i: i['filename'])
        self.datasets[dataset_name] = images_list

    def get_dataset(self, dataset):
        return self.datasets.get(dataset)

    def get_all_train_entries(self):
        print("\nTRAIN:")
        classes = {}
        number_of_objects = 0
        entries = []
        for dataset in self.datasets:
            for entry in self.datasets[dataset]["train"]:
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
        return entries

    def get_all_test_entries(self):
        print("\nTEST:")
        classes = {}
        number_of_objects = 0
        entries = []
        for dataset in self.datasets:
            for entry in self.datasets[dataset]["test"]:
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
        return entries
    
    def head_dataset(self, dataset):
        data = self.get_dataset(dataset)
        if data is None:
            return
        print("")
        print("")
        
        print(f"DATASET: {dataset}")
        
        for i, entry in enumerate(data["train"]):
            if i > 10:
                return
            print("")
            print(f"Filename: {entry.get('filename')}\nObjects: {entry.get('objects')}")
    
    def stats_dataset(self, dataset):
        data = self.get_dataset(dataset)
        if data is None:
            return
        print("")
        print("")
        
        train_length = int(len(data['train']))
        test_length = int(len(data['test']))
        length = train_length + test_length
        print(f"DATASET: {dataset}")
        images_paths = set()

        classes = {}
        number_of_objects = 0
        for entry in data["train"]:
            images_paths.add(entry["images_path"])
            number_of_objects += len(entry["objects"])
            for obj in entry["objects"]:
                if obj["class"] in classes:
                   classes[obj["class"]] += 1
                else:
                   classes[obj["class"]] = 1

        for entry in data["test"]:
            number_of_objects += len(entry["objects"])
            for obj in entry["objects"]:
                if obj["class"] in classes:
                   classes[obj["class"]] += 1
                else:
                   classes[obj["class"]] = 1

        print(f"Images paths:")
        for image_path in images_paths:
            print(f" - {image_path}")
        print(f"Number of files: {length}")
        print(f" - Train: {train_length}")
        print(f" - Test: {test_length}")
        
        print(f"Number of objects: {number_of_objects}")
        for obj_class in classes:
            print(f" - {obj_class}: {classes[obj_class]}")
    
    def print_filenames(self):
        for entry in self.get_all_test_entries():
            print(entry["filename"])

    def check_duplicates(self):
        training_set = self.get_all_train_entries()
        testing_set = self.get_all_test_entries()
        
        train_tmp = set(training_set)
        if len(train_tmp) != len(training_set):
            print("Duplicates in training set!")

        test_tmp = set(testing_set)
        if len(test_tmp) != len(testing_set):
            print("Duplicates in testing set!")

        for train_entry in training_set:
            for test_entry in testing_set:
                if train_entry["filename"] == test_entry["filename"]:
                    print("Training entry is also in test set!")
                    print(f"Train Entry: {train_entry}")
                    print(f"Test Entry: {test_entry}")

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
    
    train_test_distribution = 0.9
    classes = {"car": "1", "truck": "2", "bus": "3", "bike": "4", "person": "5", "motorbike": "6"}
    
    preparer = Prepare(train_test_distribution, datasets, classes)
    preparer.stats_dataset("other1")
    preparer.stats_dataset("other2")
    preparer.stats_dataset("other3")
    preparer.stats_dataset("self_annotated")
    for i in range(1, 13):
        name = "self_annotated" + str(i)
        preparer.stats_dataset(name)

    preparer.get_all_train_entries()
    preparer.get_all_test_entries()
    
if __name__ == '__main__':
    main()