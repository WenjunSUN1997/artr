import torch
from tools.class_evaluator import Evaluator
from tqdm import tqdm

@torch.no_grad()
def validate(model, dataloader, loss_func):
    p = []
    r = []
    f = []
    class_loss_all = []
    para_loss_all = []
    for data in tqdm(dataloader):
        label_paragraph = data['label_paragraph']
        label_classification = data['label_classification']
        label_shape = data['label_shape']
        semantic_embedding = data['semantic_embedding']
        mask = data['mask']
        output = model(text_embedding=semantic_embedding, mask=mask)
        classifi_result = output['classification']
        paragraph_logits = output['paragraph_logits']
        classifi_loss, para_loss = loss_func(label_para=label_paragraph, label_shape=label_shape,
                                             label_classi=label_classification,
                                             classification=classifi_result,
                                             paragraph_logits=paragraph_logits)
        class_loss_all.append(classifi_loss)
        para_loss_all.append(para_loss)
        try:
            gt = get_gt(data)
            hy = get_hy(output, data)
        except:
            continue

        for index in range(len(gt)):
            obj = Evaluator(pre=hy[index], label=gt[index])
            p.append(obj.get_p_value())
            r.append(obj.get_r_value())
            f.append(2*obj.get_p_value()*obj.get_r_value()
                     / (obj.get_p_value()+obj.get_r_value()))
    print('va')
    print(sum(class_loss_all) / len(class_loss_all))
    print(sum(para_loss_all) / len(para_loss_all))
    all_loss = sum(class_loss_all) / len(class_loss_all) \
               + sum(para_loss_all) / len(para_loss_all)
    # print(p)
    # print(r)
    # print(f)
    try:
        print(sum(f) / len(f))
    except:
        print('no article')
    return all_loss

def get_gt(data):
    '''
    :param data: output of the dataloader
    :return: gt[b_s, [...]]
    '''
    label_paragraph = data['label_paragraph']
    label_shape = data['label_shape']
    gt = []
    for batch_index in range(label_paragraph.shape[0]):
        row_num, col_num = label_shape[batch_index]
        label_para_real = label_paragraph[batch_index][:row_num, :col_num]
        max_indices = torch.argmax(label_para_real, dim=0)
        gt.append(max_indices.tolist())

    return gt

def get_hy(output, label_data):
    '''
    :param output: output of the model
    :param label_data:output of the dataloader
    :return: hy[b_s, [...]]
    '''
    hy = []
    class_pre = output['classification']
    para_pre = output['paragraph_logits']
    label_shape = label_data['label_shape']
    for batch_index in range(class_pre.shape[0]):
        class_pre_one_batch = class_pre[batch_index]
        para_pre_one_batch = para_pre[batch_index]
        _, para_num = label_shape[batch_index]
        _, max_index = torch.max(class_pre_one_batch, dim=1)
        row_selected = torch.where(max_index == 1)[0]
        para_pre_selected = para_pre_one_batch[row_selected, :para_num]
        para_prob = torch.softmax(para_pre_selected, dim=0)
        max_indices = torch.argmax(para_prob, dim=0)
        hy.append(max_indices.tolist())

    return hy


