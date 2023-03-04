from xml_reader import XmlProcessor
import argparse
from ast import literal_eval
import pandas as pd
import random

def create_original_csv(train_csv_path, test_csv_path):
    df = pd.DataFrame({'text_1': [],
                       'text_2': [],
                       'text_all': [],
                       'bbox_1': [],
                       'bbox_2': [],
                       'img_path_1': [],
                       'img_path_2': [],
                       'label': []})
    df.to_csv(train_csv_path, header=True)
    df.to_csv(test_csv_path, header=True)
    print('original csv done')

def create_data_random_select(annotations, csv_path):
    text_1 = []
    text_2 = []
    bbox_1 = []
    bbox_2 = []
    text_all = []
    img_path_1 = []
    img_path_2 = []
    lable = []

    for ann_index, ann in enumerate(annotations):
        reading_order_1 = ann['reading_order']
        ann_same = [ann_cell for ann_cell in annotations
                    if ann_cell['reading_order']==reading_order_1]
        ann_not_same = [ann_cell for ann_cell in annotations
                        if ann_cell['reading_order'] != reading_order_1]
        ann_same = random.sample(ann_same, min(10, len(ann_same), len(ann_not_same)))
        ann_not_same = random.sample(ann_not_same, min(10, len(ann_same), len(ann_not_same)))
        for ann_same_or_not in [ann_same, ann_not_same]:
            for ann_cell_index, ann_cell in enumerate(ann_same_or_not):
                text_1.append(ann['text'])
                text_2.append(ann_cell['text'])
                text_all.append(ann['text'] + ' '
                                + ann_cell['text'])
                bbox_1.append(ann['bbox'][0] + ann['bbox'][2])
                bbox_2.append(ann_cell['bbox'][0] + ann_cell['bbox'][2])
                img_path_1.append(ann['img_path'])
                img_path_2.append(ann_cell['img_path'])
                if ann['reading_order'] == ann_cell['reading_order']:
                    lable.append(1)
                else:
                    lable.append(0)

    data_dict = {'text_1': text_1,
                 'text_2': text_2,
                 'text_all': text_all,
                 'bbox_1': bbox_1,
                 'bbox_2': bbox_2,
                 'img_path_1': img_path_1,
                 'img_path_2': img_path_2,
                 'label': lable}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(csv_path, mode='a', header=False)

def create_data_from_anno_one_by(annotations, csv_path):
    text_1 = []
    text_2 = []
    bbox_1 = []
    bbox_2 = []
    text_all = []
    img_path_1 = []
    img_path_2 = []
    lable = []

    for ann_index, ann in enumerate(annotations):
        if ann_index+1 >= len(annotations):
            break

        text_1.append(annotations[ann_index]['text'])
        text_2.append(annotations[ann_index+1]['text'])
        text_all.append(annotations[ann_index]['text'] + ' '
                        + annotations[ann_index+1]['text'])
        bbox_1.append(annotations[ann_index]['bbox'][0] +
                      annotations[ann_index]['bbox'][2])
        bbox_2.append(annotations[ann_index+1]['bbox'][0] +
                      annotations[ann_index+1]['bbox'][2])
        img_path_1.append(annotations[ann_index]['img_path'])
        img_path_2.append(annotations[ann_index+1]['img_path'])
        if annotations[ann_index]['reading_order'] \
                            == annotations[ann_index+1]['reading_order']:
            lable.append(1)
        else:
            lable.append(0)

    data_dict = {'text_1': text_1,
                 'text_2': text_2,
                 'text_all': text_all,
                 'bbox_1': bbox_1,
                 'bbox_2': bbox_2,
                 'img_path_1': img_path_1,
                 'img_path_2': img_path_2,
                 'label': lable}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(csv_path, mode='a', header=False)

def create_data(file_list, csv_path, lang, method):
    path = {'fre': 'AS_TrainingSet_BnF_NewsEye_v2/',
            'fin': 'AS_TrainingSet_NLF_NewsEye_v2/'}
    method_dict = {'one_by':create_data_from_anno_one_by,
                   'random':create_data_random_select}
    for file_index, file_path in enumerate(file_list):
        print(file_index)
        xml_processor_obj = XmlProcessor(file_index, path[lang]+file_path)
        annotations = xml_processor_obj.get_annotation()
        method_dict[method](annotations, csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang")
    parser.add_argument("--file_path")
    parser.add_argument("--method")
    args = parser.parse_args()
    args.lang = 'fin'
    args.file_path = 'text_bbox_data'
    args.method = 'random'

    lang = args.lang
    file_path = args.file_path
    method = args.method

    train_csv_path = 'train_test_data/train_'+file_path+'_' + lang + '.csv'
    test_csv_path = 'train_test_data/test_'+file_path+'_' + lang + '.csv'
    create_original_csv(train_csv_path, test_csv_path)

    train_file_list_path = 'train_test_file_record/train_file_list_' + lang + '.txt'
    test_file_list_path = 'train_test_file_record/test_file_list_' + lang + '.txt'
    with open(train_file_list_path, 'r') as file:
        train_file_list = literal_eval(file.readlines()[0])[:]

    with open(test_file_list_path, 'r') as file:
        test_file_list = literal_eval(file.readlines()[0])[:]

    create_data(train_file_list, train_csv_path, lang, method)
    create_data(test_file_list, test_csv_path, lang, method)












