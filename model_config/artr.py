import torch

class Artr(torch.nn.Module):
    def __init__(self, num_obj_query, hidd_dim, max_len, device):
        super(Artr, self).__init__()
        self.device = device
        self.normalize = torch.nn.functional.normalize
        self.num_obj_query = num_obj_query
        self.obj_query_embedding = torch.nn.Embedding(num_obj_query, hidd_dim)
        self.ar_transformer = torch.nn.Transformer(d_model=hidd_dim, batch_first=True)

        self.classifi_linear_1 = torch.nn.Linear(in_features=hidd_dim,
                                                 out_features=2048)
        self.classifi_linear_2 = torch.nn.Linear(in_features=2048,
                                                 out_features=2)

        self.paragraph_linear_1 = torch.nn.Linear(in_features=hidd_dim,
                                                 out_features=2048)
        self.paragraph_linear_2 = torch.nn.Linear(in_features=2048,
                                                  out_features=max_len)

        self.activation_classifi = torch.nn.ReLU()
        self.activation_paragraph = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

    def get_obj_query_embedding(self, batch_size):
        obj_query_index = torch.tensor([x for x in range(self.num_obj_query)])
        obj_query_embedding = self.obj_query_embedding(obj_query_index.to(self.device))
        obj_query_embedding_batched = obj_query_embedding.repeat(batch_size, 1, 1)
        return obj_query_embedding_batched

    def forward_obj_query(self, text_embedding, mask):
        '''
        :param text_embedding: padded text embedding of one newspaper [b_s, 500, hidd_dim]
        :param mask: attention mask [b_s, 500] 0 for no_masked, 1 for masked
        :param obj_query_embedding: [b_s, num_obj_query, hidd_dim]
        :return: query_result: [b_s, num_obj_query, hidd_dim]
        '''
        obj_query_embedding_batched = self.get_obj_query_embedding(text_embedding.shape[0])
        query_result = self.ar_transformer(text_embedding,
                                     obj_query_embedding_batched,
                                     src_key_padding_mask=mask)
        return query_result

    def classification(self, query_result):
        '''
        :param query_result: output from the decoder [b_s, num_obj_query, hidd_dim]
        :return: classification result [b_s, num_obj_query, 2]
        '''
        query_result = self.normalize(query_result)
        result_classi = self.classifi_linear_1(query_result)
        result_classi = self.activation_classifi(result_classi)
        result_classi = self.classifi_linear_2(result_classi)
        classifi_result = self.softmax(result_classi)
        return classifi_result

    def paragraph_detection(self, query_result):
        '''
        :param query_result: output from the decoder [b_s, num_obj_query, hidd_dim]
        :return: classification logits [b_s, num_obj_query, max_len]
        '''
        query_result = self.normalize(query_result)
        result = self.paragraph_linear_1(query_result)
        result = self.activation_paragraph(result)
        result = self.paragraph_linear_2(result)
        paragraph_logits = self.activation_paragraph(result)
        return paragraph_logits

    def forward(self, text_embedding, mask):
        text_embedding = self.normalize(text_embedding)
        query_result = self.forward_obj_query(text_embedding, mask)
        classifi_result = self.classification(query_result)
        paragraph_logits = self.paragraph_detection(query_result)
        return {'classification': classifi_result,
                 'paragraph_logits': paragraph_logits}



