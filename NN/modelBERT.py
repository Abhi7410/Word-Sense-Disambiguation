import math
import torch
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertConfig
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
Bert_model = BertModel.from_pretrained('bert-base-uncased')
# Bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class BERT_for_WSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ranking_linear = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,batch):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output
    
    def glossSelection(self, batches):
        batch_loss = 0
        logits_list = []
        loss_fn = torch.nn.CrossEntropyLoss()
        for batch in batches:
            logits = self.ranking_linear(self.forward(batch)).squeeze(-1)
            labels = torch.max(batch[3].to(device), -1).indices.to(device).detach()
            batch_loss += loss_fn(logits, labels)
            logits_list.append(logits)
        return batch_loss, logits_list


def get_model_and_tokenizer():
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERT_for_WSD.from_pretrained('bert-base-uncased', config=config)

    if '[TGT]' not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    return model, tokenizer

