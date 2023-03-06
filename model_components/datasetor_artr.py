import os

import torch
from transformers import AutoTokenizer
from xml_reader import XmlProcessor
from model_config.tok_no_con import TokBertDiffer
from model_config.tok_con import TokBertConDiffer
from transformers import BertModel

class TextDataSeter(torch.utils.data.Dataset):
    def __init__(self, xml_path_list, lang, max_len_para, max_len_arti,semantic_dim,
                 device, model_path, model_type):
        self.xml_path_list = xml_path_list
        self.max_len_para = max_len_para
        self.max_len_arti = max_len_arti
        self.device = device
        self.lang = lang
        self.semantic_dim = semantic_dim
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        self.semantic_encoder = self._init_semantic_encoder()
        self.tokenizer = self._init_tokenizer()

    def _normalize_bbox(self, bbox):
        return [
            int(3000 * (min(bbox[0], 10000) / 10000)),
            int(5000 * (min(bbox[1], 10000) / 10000)),
            int(3000 * (min(bbox[2], 10000) / 10000)),
            int(5000 * (min(bbox[3], 10000) / 10000)),
        ]

    def _init_semantic_encoder(self):
        bert_model_dict = {'fre': BertModel.from_pretrained('camembert-base'),
                           'fin': BertModel.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')}
        semantic_encoder_dict = {'con': TokBertConDiffer(bert_model_dict[self.lang],
                                         self.semantic_dim).to(self.device),
                            'no_con': TokBertDiffer(bert_model_dict[self.lang],
                                         self.semantic_dim).to(self.device),}
        semantic_encoder = semantic_encoder_dict[self.model_type]
        semantic_encoder.load_state_dict(torch.load(self.model_path,
                                                    map_location=self.device))
        return semantic_encoder

    def _init_tokenizer(self):
        tokenizer_dict = {'fre': AutoTokenizer.from_pretrained('camembert-base'),
                          'fin': AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')}
        return tokenizer_dict[self.lang]

    def get_text_bbox_ro_from_xml(self, xml_path):
        annotation_list = XmlProcessor(1, xml_path).get_annotation()
        reading_order = [x['reading_order'] for x in annotation_list]
        text = [x['text'] for x in annotation_list]
        bbox = [self._normalize_bbox(x['bbox'][0] + x['bbox'][2])
                for x in annotation_list]
        x_pos = [[int((x[0]+x[2]) / 2)] for x in bbox]
        y_pos = [[int((x[1]+x[3]) / 2)] for x in bbox]
        return {'reading_order': reading_order,
                'text': text,
                'x_pos': x_pos,
                'y_pos': y_pos}

    def create_gt(self, reading_order):
        d = {}
        for i, v in enumerate(reading_order):
            if v not in d:
                d[v] = []
            d[v].append(i)

        article_list = list(d.values())
        label = torch.zeros(len(article_list), len(reading_order))
        for article_index, article in enumerate(article_list):
            for text_index in article:
                label[article_index][text_index] = 1

        return torch.tensor(label).to(self.device)

    def padding_semantic(self, semantic_embedding_list):
        padding_length = self.max_len_para-len(semantic_embedding_list)
        mask = [False]*len(semantic_embedding_list) + [True]*padding_length
        mask = torch.tensor(mask).to(self.device)
        tensor_pad = torch.zeros(padding_length, self.semantic_dim).to(self.device)
        semantic_embedding_padded = torch.cat((semantic_embedding_list,tensor_pad), dim=0)
        return mask, semantic_embedding_padded

    def padding_lable(self, label):
        padded_label = torch.zeros(self.max_len_arti, self.max_len_para)
        padded_label[:label.shape[0], :label.shape[1]] = label
        label_classifi = torch.zeros(self.max_len_arti, 2)
        label_classifi[:, :1] = 1
        label_classifi[:label.shape[0], 0] = 0
        label_classifi[:label.shape[0], 1] = 1
        return torch.tensor(label.shape), padded_label.to(self.device), label_classifi.to(self.device)

    def create_semantic_embdeeing(self, text:list, x, y):
        semantic_embedding_list = []
        x = torch.tensor(x).to(self.device)
        y = torch.tensor(y).to(self.device)
        for index in range(len(text)):
            output_toenizer = self.tokenizer([text[index]], max_length=512,
                                               truncation=True,
                                               padding='max_length',
                                              return_tensors = 'pt')
            output_toenizer = output_toenizer.to(self.device)

            _, _, semantic_embedding, _ = self.semantic_encoder(output_toenizer,
                                                                output_toenizer,
                                                                output_toenizer,
                                                                x[index], y[index],
                                                                x[index], y[index])
            semantic_embedding_list.append(semantic_embedding.tolist())

        return torch.tensor(semantic_embedding_list).squeeze(1).to(self.device)

    def __len__(self):
        return len(self.xml_path_list)

    def __getitem__(self, item):
        '''
        :param item:
        :return: label:[article_num, paragraph_num], 0 for the para not in the article
                semantic_embedding:[max_len_para, semantic_dim], the semantic embedding for
                                    each paragraph, 0 for padding
                mask: [0/1*max_len_para]: 0 for not padded
        '''
        data = self.get_text_bbox_ro_from_xml(self.xml_path_list[item])
        label = self.create_gt(data['reading_order'])
        shape_label, label_padded,  label_classification= self.padding_lable(label)
        semantic_embedding = self.create_semantic_embdeeing(data['text'],
                                                            data['x_pos'],
                                                            data['y_pos']).detach()
        mask, semantic_embedding_padded = self.padding_semantic(semantic_embedding)
        return {'label_paragraph': label_padded,
                 'label_classification': label_classification,
                 'label_shape': shape_label,
                 'semantic_embedding': semantic_embedding_padded,
                 'mask': mask}

# if __name__ == "__main__":
#     xml_path_list = os.listdir('../AS_TrainingSet_NLF_NewsEye_v2')
#     xml_path_list = ['../AS_TrainingSet_NLF_NewsEye_v2/'+x for x in xml_path_list if 'xml' in x]
#     x = TextDataSeter(xml_path_list=xml_path_list, lang='fin', max_len_para=500, max_len_arti=30, semantic_dim=768,
#                  device='cuda:0', model_path='../model_zoo/TokBertDiffer_fin_4_192000.pth', model_type='no_con')
#     a = x[0]
#     print(a)



