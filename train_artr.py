import torch
from model_components.hungary_loss import HungaryLoss
from model_components.validator_artr import validate
from model_config.artr import Artr
import argparse
from model_components import dataloader_artr
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def train_model(lang, max_len_para, max_len_arti,semantic_dim,
                 device, model_path, model_type):
    torch.autograd.detect_anomaly()
    LR = 5e-5
    epoch = 10000
    class_loss_all = []
    para_loss_all = []

    train_dataloader = dataloader_artr.get_dataloader(goal='train', lang=lang,
                                                      max_len_para=max_len_para,
                                                      max_len_arti=max_len_arti,
                                                      semantic_dim=semantic_dim,
                                                      device=device,
                                                      model_path=model_path,
                                                      model_type=model_type,
                                                      batch_size=batch_size)

    test_dataloader = dataloader_artr.get_dataloader(goal='test', lang=lang,
                                                      max_len_para=max_len_para,
                                                      max_len_arti=max_len_arti,
                                                      semantic_dim=semantic_dim,
                                                      device=device,
                                                      model_path=model_path,
                                                      model_type=model_type,
                                                      batch_size=batch_size)

    model = Artr(num_obj_query=max_len_arti, hidd_dim=semantic_dim,
                 max_len=max_len_para, device=device)
    model.to(device=device)
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=0.2, patience=3, verbose=True)
    loss_func = HungaryLoss(device=device)
    for epoch_num in range(epoch):
        for step_index, data in enumerate(train_dataloader):
            label_paragraph = data['label_paragraph']
            label_classification = data['label_classification']
            label_shape = data['label_shape']
            semantic_embedding = data['semantic_embedding']
            mask = data['mask']

            output = model(text_embedding=semantic_embedding , mask=mask)
            classifi_result = output['classification']
            paragraph_logits = output['paragraph_logits']

            classifi_loss, para_loss = loss_func(label_para=label_paragraph, label_shape=label_shape,
                             label_classi=label_classification,
                             classification=classifi_result,
                             paragraph_logits=paragraph_logits)
            class_loss_all.append(classifi_loss)
            para_loss_all.append(para_loss)
            loss_all = classifi_loss + para_loss
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        print(epoch_num)
        print(sum(class_loss_all) / len(class_loss_all))
        print(sum(para_loss_all) / len(para_loss_all))
        class_loss_all = []
        para_loss_all = []
        val_loss, f = validate(model=model, dataloader=test_dataloader,
                            loss_func=loss_func)
        scheduler.step(f)
        if epoch_num % 300 == 0:
            torch.save(model.state_dict(),
                       'model_zoo/artr_'+lang+str(epoch_num)+'.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='fin')
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--model_type", default='no_con')
    parser.add_argument("--max_len_para", default=400)
    parser.add_argument("--max_len_arti", default=115)
    parser.add_argument("--semantic_dim", default=768)
    parser.add_argument("--model_path", default='model_zoo/TokBertDiffer_fin_4_192000.pth')
    args = parser.parse_args()
    print(args)
    lang = args.lang
    batch_size = int(args.batch_size)
    device = args.device
    model_type = args.model_type
    semantic_dim = args.semantic_dim
    model_path = args.model_path
    max_len_para = args.max_len_para
    max_len_arti = args.max_len_arti

    train_model(lang=lang, max_len_para=max_len_para, max_len_arti=max_len_arti,
                semantic_dim=semantic_dim, device=device, model_path=model_path,
                model_type=model_type)





