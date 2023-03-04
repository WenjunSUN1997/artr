import torch
from model_components.model_validator import val_model
from transformers import BertModel
from model_config.tok_con import TokBertConDiffer
from model_config.tok_no_con import TokBertDiffer
from tqdm import tqdm
from model_components.my_loss_func import LossFunc
from model_components.dataloader_creator import get_dataloader
import argparse

def train_model(model, LR, loss_function, train_dataloader,
                test_dataloader, validator, device,lang):
    step = 0
    epoch = 1000
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    loss_all = []
    print('models loaded')

    for epoch_num in range(epoch):
        if epoch_num % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5

        for data in tqdm(train_dataloader):
            text_1 = data['text_1']
            text_2 = data['text_2']
            text_all = data['text_all']
            label = data['label'].to(device)

            x_1 = data['x_1'].to(device)
            y_1 = data['y_1'].to(device)
            x_2 = data['x_2'].to(device)
            y_2 = data['y_2'].to(device)

            text_1_cls, text_2_cls, sematic_1_final, sematic_2_final = \
                model(text_1, text_2, text_all, x_1, y_1, x_2, y_2)
            loss = loss_function(text_1_cls, text_2_cls, sematic_1_final,
                                 sematic_2_final, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step+=1
            loss_all.append(loss.item())

            if step % 100 == 0:
                print(epoch_num, step)
                print(sum(loss_all) / len(loss_all))
                loss_all = []

            if step % 6000 == 0:
                torch.save(model.state_dict(), 'model_zoo/' + type(model).__name__ +
                           '_' + lang + '_' + str(epoch_num) + '_' + str(step) + '.pth')
                validator(model, test_dataloader, loss_function, device, lang)

        torch.save(model.state_dict(), 'model_zoo/' + type(model).__name__ +
                           '_' +lang + '_' + str(epoch_num) + '_' + str(step) + '.pth')
        validator(model, test_dataloader, loss_function, device, lang)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang")
    parser.add_argument("--batch_size")
    parser.add_argument("--device")
    parser.add_argument("--model_type")
    parser.add_argument("--data_type")
    args = parser.parse_args()
    print(args)

    # args.lang = 'fre'
    # args.device = 'cuda:0'
    # args.model_type = 'con'
    # args.batch_size = 1
    # args.data_type = 'random'

    lang = args.lang
    device = args.device
    model_type = args.model_type
    batch_size = int(args.batch_size)
    data_type = args.data_type

    hidd_dim_dict = {'fre': 768, 'fin': 768}
    bert_model_dict = {'fre': BertModel.from_pretrained('camembert-base'),
                       'fin': BertModel.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')}
    model_dict = {'con': TokBertConDiffer(bert_model_dict[lang],
                                          hidd_dim_dict[lang]).to(device),
                  'no_con': TokBertDiffer(bert_model_dict[lang],
                                          hidd_dim_dict[lang]).to(device)
                  }

    LR = 5e-5
    loss_func = LossFunc(ratio_cls=0.6, ratio_model_output=0.4)
    train_dataloader = get_dataloader(goal='train', lang=lang, data_type=data_type,
                                      batch_size=batch_size, device=device)

    test_dataloader = get_dataloader(goal='test', lang=lang, data_type=data_type,
                                      batch_size=batch_size, device=device)
    model = model_dict[model_type]
    validator = val_model

    train_model(model=model, LR=LR, loss_function=loss_func,
                train_dataloader = train_dataloader,
                test_dataloader = test_dataloader,
                validator=validator, device=device,lang=lang)

