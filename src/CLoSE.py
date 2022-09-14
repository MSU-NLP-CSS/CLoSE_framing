import torch
from transformers import BertModel, RobertaModel

class CLoSE(torch.nn.Module):
    """
    Arguments for the CLoSE initialization are:
     - model_name: encoder name,     type = string
                   'roberta-base', 'bert-base-uncased', or 'bert-base-cased'
     - dropout: droupout rate,       type = float
     - gpu_id: GPU id for assigning, type = int
    """

    def __init__(self, model_name, dropout=0.5, gpu_id=0):

        super(CLoSE, self).__init__()

        label_size = 2
        hidden_size = 768

        if 'roberta' in model_name:
            self.bert = RobertaModel.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_size, label_size)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask, device, batch_size):

        token_embeddings, _ = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        anchor_idx = torch.tensor([3*i for i in range(batch_size)]).to(device)
        dropout_output = self.dropout(torch.index_select(sentence_embeddings, 0, anchor_idx))
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return sentence_embeddings, final_layer