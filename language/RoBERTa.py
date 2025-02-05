# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import random
import os
import numpy as np

from torch.autograd import Variable
from utils import *

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import logging    # first of all import the module
from datetime import datetime
from transformers import RobertaConfig, RobertaForSequenceClassification
from modeling_roberta_relu import RobertaForSequenceClassification as RobertaForSequenceClassificationRelu

# %%
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(1)
def get_savefilename(foldername,filename):
    logfilename = os.path.join(foldername, filename)
    return logfilename

class TextClassifier(nn.Module):
    def __init__(self, args, foldername ):
        super(TextClassifier, self).__init__()
        if args.model_name=='roberta-scratch':
            configuration = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=args.num_labels).config
            configuration.num_hidden_layers = args.num_layers
            configuration.num_attention_heads = args.num_head
            configuration.hidden_size = args.hidden_dim
            configuration.attention_probs_dropout_prob = args.dropout
            configuration.hidden_dropout_prob = args.dropout
            if args.roberta_act == 'relu':
                self.model = RobertaForSequenceClassificationRelu(configuration)
            else:
                self.model = RobertaForSequenceClassification(configuration)
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            self.model =AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels) 
            configuration = self.model.config


        self.embedding_noise_variance = args.embedding_noise_variance
        self.sensitivity_method = args.sensitivity_method
        self.config = configuration
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.device = args.cudaname
        # self.fc = nn.Linear(self.model.config.hidden_size, 3)  # FC layer
        self.criterion = torch.nn.CrossEntropyLoss()#ignore_index=0
        self.softmax = torch.nn.Softmax(dim=1)
        if args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('optimizer not implemented')
        #self.optimizer = torch.optim.Adam(self.model.parameters(),  lr= args.lr ,  betas=(0, 0.9)  )
        self.scheduler =torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=args.gamma)
        self.epochs = args.epochs
        self.report_number = args.report_num_points
        self.replace_size = args.replace_size
        self.sensitivity_2dlist_report = []
        self.sensitivity_1dlist_report = []
        self.extra_validation_loss_report = []
        self.extra_validation_acc_report = []
        self.train_loss_report = []
        self.validation_loss_report = []
        self.log_foldername = foldername
        self.validation_acc_report = []

    def forward(self,x,x_att):
        output = self.model(x,attention_mask=x_att)
        logits = output.logits
        # logits = self.softmax(logits) #not sure whether we need this line to compute loss
        return logits
    
    def forward_withembedding(self,x_embedding,x_att):#to do the Embedding Label-Sensitivity, we need add noise to the word_embedding
        output = self.model(inputs_embeds=x_embedding,attention_mask=x_att)
        logits = output.logits
        # logits = self.softmax(logits) #not sure whether we need this line to compute loss
        return logits
        
    
    def loss(self,logits,labels):
        loss = self.criterion(logits, labels)
        return loss
    def validation(self,valid_dataloader, replaced_dataloader,embedding_sens_eval_dataloader, epoch,device,save=False,extra_valid_dataloader=None):
        logger =  logging.getLogger('training')
        self.model.eval()
        all_val_acc = []
        all_val_loss = []
        prediction_val = []
        extra_validation_all_val_acc = []
        extra_validation_all_val_loss = []
        prediction_extra_val = []
        with torch.no_grad():
            for step,batch in enumerate(valid_dataloader):
                text, text_mask,labels = batch
                text = text.to(device)
                text_mask = text_mask.to(device)
                labels = labels.to(device)
                logits = self.forward(text,text_mask)
                loss = self.loss(logits,labels)
                predict = torch.argmax(logits,dim=1)
                accuracy = accuracy_score(labels, predict)
                prediction_val.append(predict)
                all_val_loss.append(loss)
                all_val_acc.append(accuracy)
            if extra_valid_dataloader != None:
                for step,batch in enumerate(extra_valid_dataloader):
                    text, text_mask,labels = batch
                    text = text.to(device)
                    labels = labels.to(device)
                    text_mask = text_mask.to(device)
                    logits = self.forward(text,text_mask)
                    loss = self.loss(logits,labels)
                    predict = torch.argmax(logits,dim=1)
                    accuracy = accuracy_score(labels, predict)
                    prediction_extra_val.append(predict)
                    extra_validation_all_val_loss.append(loss)
                    extra_validation_all_val_acc.append(accuracy)
                prediction_extra_val = torch.cat(prediction_extra_val, dim=0)
                prediction_val = torch.cat(prediction_val, dim=0)
        if(self.sensitivity_method == 'word'):
            sensitivity = word_label_sensitivity(replaced_dataloader, valid_dataloader, self, device, self.replace_size)
            mean_sensitivity = -1
        elif self.sensitivity_method =='embedding':
            sensitivity, mean_sensitivity = embedding_label_sensitivity(embedding_sens_eval_dataloader, self, self.embedding, device, self.replace_size, self.embedding_noise_variance)
        elif self.sensitivity_method =='sentence_embedding':# we use a subset of training data to test the sensitivity
            sensitivity, mean_sensitivity = sentence_embedding_label_sensitivity(embedding_sens_eval_dataloader, self, self.embedding, device, self.replace_size, self.embedding_noise_variance)
        self.validation_loss_report.append(sum(all_val_loss)    / len(all_val_loss) )
        self.validation_acc_report.append(sum(all_val_acc) / len(all_val_acc))
        self.extra_validation_loss_report.append(sum(extra_validation_all_val_loss)    / len(extra_validation_all_val_loss) )
        self.extra_validation_acc_report.append(sum(extra_validation_all_val_acc) / len(extra_validation_all_val_acc))
        if self.sensitivity_method!= 'none':
            self.sensitivity_2dlist_report.append(sensitivity)
            self.sensitivity_1dlist_report.append(mean_sensitivity)
            logger.info(f"sensitivity: {sensitivity}")
            logger.info(f"sensitivity mean: {mean_sensitivity :.4f}")
        logger.info(f"Validation Loss:  {sum(all_val_loss)    / len(all_val_loss)    :.4f}")
        logger.info(f"Validation Accuracy: {sum(all_val_acc) / len(all_val_acc):.4f}")
        differences = torch.sum(prediction_val != prediction_extra_val).item()
        percentage_diff = (differences / prediction_val.numel()) 
        logger.info(f"Percentage difference between extra_validation_all_val_acc and all_val_acc: {percentage_diff:.2f}")
        logger.info(f"extra_validation Loss:  {sum(extra_validation_all_val_loss)    / len(extra_validation_all_val_loss)    :.4f}")
        logger.info(f"extra_validation Accuracy: {sum(extra_validation_all_val_acc) / len(extra_validation_all_val_acc):.4f}")
        if save:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'acc': sum(all_val_acc) / len(all_val_acc),}, './checkpoints/'+str(epoch)+'_model.pt')
        self.model.train()
    def train(self,train_dataloader,valid_dataloader,embedding_sens_eval_dataloader,replaced_dataloader, device,extra_valid_dataloader=None):
        logger =  logging.getLogger('training')
        logger.info(f'model config:{self.config}')
        self.validation(valid_dataloader, replaced_dataloader,embedding_sens_eval_dataloader, 0,device,False,extra_valid_dataloader=extra_valid_dataloader)
        self.model.train()
        report_counter  = 0
        total_counter = 0
        epoch_train_loss = 0.0
        batch_train_loss = 0.0
        batch_loss_counter = 0
        for epoch in range(self.epochs):
            epoch_train_loss = 0.0
            for step,batch in enumerate(train_dataloader):
                report_counter += len(batch[0])
                total_counter += len(batch[0])
                input_ids = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
                input_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
                labels = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)    
                self.optimizer.zero_grad()
                logits = self.forward(input_ids,input_attn)
                loss = self.loss(logits,labels)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                batch_train_loss += loss.item()
                batch_loss_counter += 1


                if report_counter > self.report_number:
                    logger.info(f"Average training loss for each batch:{batch_train_loss/(batch_loss_counter)} ")
                    self.train_loss_report.append(batch_train_loss/(batch_loss_counter+1))
                    report_counter  = 0
                    batch_loss_counter = 0
                    batch_train_loss = 0
                    logger.info(f'======total trained data counter: {total_counter}======')
                    logger.info(f'report_counter hit { self.report_number}, will do validation')
                    # Validation
                    self.validation(valid_dataloader, replaced_dataloader,embedding_sens_eval_dataloader,epoch,device,False,extra_valid_dataloader=extra_valid_dataloader)

                    if self.sensitivity_method == 'sentence_embedding':
                        temp1 = torch.tensor(self.sensitivity_1dlist_report, device = 'cpu')
                    else:
                        temp1 = torch.tensor(self.sensitivity_2dlist_report, device = 'cpu')
                    temp2= torch.tensor(self.train_loss_report, device = 'cpu')
                    temp3 = torch.tensor(self.validation_loss_report, device = 'cpu')
                    temp4 = torch.tensor(self.validation_acc_report, device = 'cpu')
                    temp5 = torch.tensor(self.extra_validation_loss_report, device = 'cpu')
                    temp6 = torch.tensor(self.extra_validation_acc_report, device = 'cpu')
                    np.save(get_savefilename(self.log_foldername,'sensitivity'), temp1)
                    np.save(get_savefilename(self.log_foldername,'trainloss'),temp2)
                    np.save(get_savefilename(self.log_foldername,'validationloss'), temp3)
                    np.save(get_savefilename(self.log_foldername,'validationacc'), temp4)
                    np.save(get_savefilename(self.log_foldername,'extra_validationloss'), temp5)
                    np.save(get_savefilename(self.log_foldername,'extra_validationacc'), temp6)

            self.scheduler.step()
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_train_loss/(step+1):.4f}")

            
   




# %%
