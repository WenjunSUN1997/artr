from torch.utils.data.dataloader import DataLoader
from ast import literal_eval
from model_components.datasetor_artr import TextDataSeter

def get_dataloader(goal, lang, max_len_para, max_len_arti, semantic_dim,
                 device, model_path, model_type, batch_size):
    xml_prefix_dict = {'fin': 'AS_TrainingSet_NLF_NewsEye_v2/',
                       'fre': 'AS_TrainingSet_BnF_NewsEye_v2/'}
    with open('train_test_file_record/'+goal+'_file_list_'+lang+'.txt', 'r') as file:
        file_list = literal_eval(file.readlines()[0])
    file_list = [xml_prefix_dict[lang]+x for x in file_list]

    dataset = TextDataSeter(xml_path_list=file_list, lang=lang,
                            max_len_para=max_len_para,
                            max_len_arti=max_len_arti,
                            semantic_dim=semantic_dim,
                            device=device,
                            model_path=model_path,
                            model_type=model_type)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

