import torch
from scipy.optimize import linear_sum_assignment
import numpy as np

class HungaryLoss(torch.nn.Module):
    def __init__(self, no_article_weight=0.7, device='cuda:0'):
        super(HungaryLoss, self).__init__()
        self.no_article_weight = no_article_weight
        self.classi_loss = []
        self.para_loss = []
        self.device = device

    def create_label_mask(self, label_para, label_shape):
        '''
        :param label_para: [b_s, max_len_arti, max_len_para]
        :param label_shape: [b_s, 2] the shape the real label
        :return: mask_list [b_s, [max_len_arti,max_len_para]], 1 for remaining
        '''
        b_s, max_len_arti, max_len_para = label_para.shape
        mask = []
        for batch_index in range(b_s):
            len_para = label_shape[batch_index][1]
            mask_temp = torch.zeros(max_len_arti, max_len_para)
            mask_temp[:, :len_para] = 1
            mask.append(mask_temp.to(self.device))

        return mask

    @torch.no_grad()
    def match(self, label_para, label_shape, label_classi,
                        classification, paragraph_logits):
        '''
        :param label_para: [b_s, max_len_arti, max_len_para], 1 for in the article
        :param label_classi: [b_s, max_len_arti, 2]
        :param label_shape: [b_s, 2] the shape the real label
        :param classification: [b_s, max_len_arti, 2]
        :param paragraph_logits: [b_s, max_len_arti, max_len_para]
        :return:
        '''
        final_loss = []
        paragraph_prob = paragraph_logits.softmax(1)
        batch_size = label_para.shape[0]
        row_list = []
        column_list = []
        for batch_index in range(batch_size):
            article_num, para_num = label_shape[batch_index]
            label_para_real = label_para[batch_index][:, :para_num]
            paragraph_prob_real = paragraph_prob[batch_index][:, :para_num]
            paragraph_loss = torch.cdist(paragraph_prob_real,
                                              label_para_real,
                                              p=1)
            classi_loss = torch.cdist(classification[batch_index],
                                           label_classi[batch_index],
                                           p=1)
            all_loss_cell = (paragraph_loss + classi_loss).detach().cpu()
            row_ind, col_ind = linear_sum_assignment(all_loss_cell)
            row_list.append(row_ind)
            column_list.append(col_ind)
            final_loss.append(all_loss_cell[row_ind, col_ind].sum())

        return row_list, column_list

    def forward(self, label_para, label_shape, label_classi,
                        classification, paragraph_logits):
        '''
        :param label_para: [b_s, max_len_arti, max_len_para], 1 for in the article
        :param label_classi: [b_s, max_len_arti, 2]
        :param label_shape: [b_s, 2] the shape the real label
        :param classification: [b_s, max_len_arti, 2]
        :param paragraph_logits: [b_s, max_len_arti, max_len_para]
        :return: loss
        '''
        b_s, max_len_arti, max_len_para = label_para.shape
        mask = self.create_label_mask(label_para, label_shape)
        mask = torch.cat(mask, dim=0)
        row_list, column_list = self.match(label_para, label_shape, label_classi,
                        classification, paragraph_logits)
        label_para = torch.stack([label_para[x][column_list[x]]
                                  for x in range(b_s)], dim=0).view(b_s * max_len_arti, -1)
        label_classi = torch.stack([label_classi[x][column_list[x]]
                                  for x in range(b_s)], dim=0).view(b_s * max_len_arti, -1)
        paragraph_prob = paragraph_logits.softmax(1).view(b_s * max_len_arti, -1)
        classification = classification.view(b_s * max_len_arti, -1)
        para_loss = torch.sum(torch.abs(paragraph_prob-label_para)*mask)
        class_loss = torch.sum(torch.abs(classification-label_classi))

        return class_loss, para_loss
