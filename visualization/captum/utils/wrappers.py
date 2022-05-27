import torch
import torch.nn as nn


# RobertaForSequenceClassification
class RobertaClfWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # Predict Classification Label
    def predict(self,input_ids, input_embeds = None):
        # return self.model(inputs)[0]
        if input_embeds is not None:
            logits = self.model(input_embeds = input_embeds).logits
        else:
            logits = self.model(input_ids).logits
        preds = torch.argmax(logits, axis=1) 
        return preds

    # input_ids as forward input doesn't work with deeplift -> can't calculate gradient
    # def forward(self, inputs):
    #     logits = self.predict(inputs)
    #     # return torch.softmax(preds, dim = 1)[0][1].unsqueeze(-1)
    #     return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)

    def forward(self, input_embeds, prob_label = 1):
        # inputs: Roberta Embedding
        # prob_label: Class of which to report probability as score
        ##### Receive Embeddings as input!!
        logits = self.model(inputs_embeds = input_embeds).logits
        return torch.softmax(logits, dim=1)[:, prob_label].unsqueeze(1)

    def get_embeddings(self, input_ids):
        return self.model.roberta.embeddings(input_ids)

# Embedding Layers
# Bert: bert
# RoBERTa: roberta
# GPT2: wte

class CLFWrapper(nn.Module):
    def __init__(self, model, pretrained_model = "roberta", embedding_layer = "embeddings"):
        super().__init__()
        self.model = model
        self.embedding_layer = getattr(getattr(self.model, pretrained_model), embedding_layer)

    def calc_logits(self, inputs_embeds, input_ids = None):
        if input_ids is not None:
            logits = self.model(inputs_embeds = self.embedding_layer(input_ids)).logits
        else:
            logits = self.model(inputs_embeds = inputs_embeds).logits
            
        return logits
    
    # Predict Classification Label
    def predict(self, input_ids = None, inputs_embeds = None, prob_label = 1):
        logits = self.calc_logits(inputs_embeds, input_ids = input_ids)

        # Predicted Class
        pred = torch.argmax(logits, axis=1) 
        # Softmax Probabilities
        prob = torch.softmax(logits, dim=1)[:, prob_label].unsqueeze(1)
        return pred, prob
    
    def forward(self, inputs_embeds, input_ids = None, target_label = 1):
    # def forward(self, input_ids, input_embeds = None):
        # inputs: Roberta Embedding
        # prob_label: Class of which to report probability as score
        logits = self.calc_logits(inputs_embeds, input_ids =input_ids)
            
        label_prob = torch.softmax(logits, dim=1)[:, target_label].unsqueeze(1)
        return label_prob
        # return logits

    def get_embeddings(self, input_ids):
        return self.embedding_layer(input_ids)