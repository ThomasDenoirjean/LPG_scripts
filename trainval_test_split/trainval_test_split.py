"""
TODO : Create docstrings !!!
TODO : clean the code
TODO : mettre une seed pour fixer le trucs random
TODO : rajotuer des arguments en cli pour nombre mini de pictures dans le train et la seed ?
TODO : faire un système qui va rajouter un tag mislabeled à l'inférence avec le modèle dans le script qui fera son évaluation
"""

import xml.etree.ElementTree as ET
import random
import copy
import csv
import argparse


parser = argparse.ArgumentParser(description='Split XML dataset into trainval and test sets.')
parser.add_argument('input_file', type=str, help='Name of your project')
parser.add_argument('--min_spec_in_trainval', type=int, default=10, help='Minimum number of specimens in a species to consider it')
parser.add_argument('--trainval_test_split', type=float, default=0.2, help='Fraction of data to keep in the test dataset')
args = parser.parse_args()

input_file = f'{args.input_file}.xml'

min_spec_in_trainval = args.min_spec_in_trainval ## TODO a tester sur un jeu de données adéquat
trainval_test_split = args.trainval_test_split

tree = ET.parse(input_file)
root = tree.getroot()

labels = root.findall('images/image/classifications/classification/code')
labels = [label.text for label in labels]

species_occu_dict = {}

for idx, label in enumerate(labels):
    if species_occu_dict.get(label, False):
        list_ = species_occu_dict[label]
        list_.append(idx)
        species_occu_dict[label] = list_
    else:
        species_occu_dict.update({label: [idx]})

for label in labels:
    assert labels.count(label) == len(species_occu_dict[label])

sorted_dict = species_occu_dict.copy()

for key in species_occu_dict.keys():
    if len(sorted_dict[key])*(1-trainval_test_split) < min_spec_in_trainval:
        sorted_dict.pop(key, None)

trainval_test_specimens_dict = {}

for key in sorted_dict.keys():
    idxs = sorted_dict[key]

    specimens_number = len(idxs)

    test_specimens_number = round(specimens_number*trainval_test_split)

    test_idx = random.sample(idxs, test_specimens_number)

    trainval_test_specimens_dict.update({
        key: {
            'trainval': list(set(idxs) - set(test_idx)),
            'test': test_idx
        }
    })

trainval_idx = []
test_idx = []

for key in trainval_test_specimens_dict.keys():
    trainval_idx += trainval_test_specimens_dict[key]['trainval']
    test_idx += trainval_test_specimens_dict[key]['test']

print(f'Length of the trainval set: {len(trainval_idx)}, and test set: {len(test_idx)}')

def split_xml_by_indexes(input_file, output1, output2, output3, trainval_idx, test_idx):
    tree = ET.parse(input_file)
    root = tree.getroot()

    taxons = root.find("taxons")
    tags = root.find("tags")
    processingInfo = root.find("processingInfo")
    settings = root.find("settings")
    images = root.find("images")

    def create_base_root():
        new_root = ET.Element(
            "project",
            {
                "version": "dev",
                "root": r"D:\Thomas Denoirjean\Fondation Model"
            }
        )
        new_root.append(copy.deepcopy(taxons))
        new_root.append(copy.deepcopy(tags))
        new_root.append(copy.deepcopy(processingInfo))
        new_root.append(copy.deepcopy(settings))
        return new_root

    root1 = create_base_root()
    root2 = create_base_root()

    images1 = ET.SubElement(root1, "images")
    images2 = ET.SubElement(root2, "images")

    img_list = images.findall("image")

    for i in trainval_idx:
        if i < len(img_list):
            images1.append(copy.deepcopy(img_list[i]))

    for i in test_idx:
        if i < len(img_list):
            images2.append(copy.deepcopy(img_list[i]))

    ET.ElementTree(root1).write(output1, encoding="utf-8", xml_declaration=True)
    ET.ElementTree(root2).write(output2, encoding="utf-8", xml_declaration=True)
    ET.ElementTree(root2).write(output3, encoding="utf-8", xml_declaration=True)


split_xml_by_indexes(
    input_file=input_file,
    output1="trainval.xml",
    output2="test.xml",
    output3="test_backup.xml",
    trainval_idx=trainval_idx,
    test_idx=test_idx
)

picture_IDs = root.findall('images/image/source/GUID')
picture_IDs = [id.text for id in picture_IDs]

test_picture_IDs = [picture_IDs[i] for i in test_idx]
test_labels = [labels[i] for i in test_idx]

with open("test_labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["picture_guid", "true_label"])

    for id, label in zip(test_picture_IDs, test_labels):
        writer.writerow([id, label])

print('Trainval and test datasets created !')