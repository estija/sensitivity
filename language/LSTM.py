import random
import os
import numpy as np
import torch
import torch.nn as nn
from utils import *
from torch.autograd import Variable 

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

class LSTMTextClassifier(nn.Module):
    def __init__(self, args, foldername):
        super(LSTMTextClassifier, self).__init__()
        
        self.embedding = nn.Sequential(
            nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=args.pad_idx),
            nn.LayerNorm(args.hidden_dim)
        )
        self.model = nn.LSTM(args.embedding_dim, args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, batch_first=True)
        self.fc = nn.Linear(args.hidden_dim, args.num_labels)
        self.dropout = nn.Dropout(args.dropout)
        self.sensitivity_method = args.sensitivity_method
        self.epochs = args.epochs
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.device = args.cudaname
        self.criterion = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        if args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('optimizer not implemented')
        #self.optimizer = torch.optim.Adam(self.model.parameters(),  lr= args.lr ,  betas=(0, 0.9)  )
        self.scheduler =torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=args.gamma)
        self.report_number = args.report_num_points
        self.replace_size = args.replace_size
        self.sensitivity_2dlist_report = []
        self.sensitivity_1dlist_report = []
        
        self.train_loss_report = []
        self.validation_loss_report = []
        self.extra_validation_loss_report = []
        self.extra_validation_acc_report = []
        self.log_foldername = foldername
        self.validation_acc_report = []
        self.embedding_noise_variance = args.embedding_noise_variance

    def forward(self, text,text_mask):
        logger =  logging.getLogger('training')
        logger.debug(f"text.shape:{text.shape}")
        embedded = self.embedding(text)
        logger.debug(f"embedded.shape:{embedded.shape}")
        logger.debug(f"text_mask.shape:{text_mask.shape}")
        text_lengths = torch.sum(text_mask,dim=1)
        text_lengths = text_lengths.cpu()
        text_lengths[text_lengths <= 0] = 1
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        # logger.debug(f"packed_embedded shape:{packed_embedded.shape}")
        h_0 = torch.zeros((self.num_layers,embedded.shape[0], self.hidden_dim),device=self.device)#hidden state
        c_0 = torch.zeros((self.num_layers,embedded.shape[0], self.hidden_dim),device=self.device) #internal state
        logger.debug(f"h_0.shape:{h_0.shape}")
        logger.debug(f"c_0.shape:{c_0.shape}")
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.model(packed, (h_0, c_0)) 
        final_hidden_state = hn[-1,:, :]
        logits = self.fc(final_hidden_state)
        return logits
    def forward_withembedding(self,embedded,text_mask):
        logger =  logging.getLogger('training')
        logger.debug(f"embedded.shape:{embedded.shape}")
        logger.debug(f"text_mask.shape:{text_mask.shape}")
        text_lengths = torch.sum(text_mask,dim=1)
        text_lengths = text_lengths.cpu()
        text_lengths[text_lengths <= 0] = 1
        h_0 = torch.zeros((self.num_layers,embedded.shape[0], self.hidden_dim),device=self.device)#hidden state
        c_0 = torch.zeros((self.num_layers,embedded.shape[0], self.hidden_dim),device=self.device) #internal state
        logger.debug(f"h_0.shape:{h_0.shape}")
        logger.debug(f"c_0.shape:{c_0.shape}")
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.model(packed, (h_0, c_0)) 
        final_hidden_state = hn[-1,:, :]
        logits = self.fc(final_hidden_state)
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
        
        if self.sensitivity_method!= 'none':
            self.sensitivity_2dlist_report.append(sensitivity)
            self.sensitivity_1dlist_report.append(mean_sensitivity)
            logger.info(f"sensitivity: {sensitivity}")
            logger.info(f"sensitivity mean: {mean_sensitivity :.4f}")
        self.validation_loss_report.append(sum(all_val_loss)    / len(all_val_loss) )
        self.validation_acc_report.append(sum(all_val_acc) / len(all_val_acc))
        self.extra_validation_loss_report.append(sum(extra_validation_all_val_loss)    / len(extra_validation_all_val_loss) )
        self.extra_validation_acc_report.append(sum(extra_validation_all_val_acc) / len(extra_validation_all_val_acc))
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
        self.validation(valid_dataloader,replaced_dataloader,embedding_sens_eval_dataloader, 0,device,False,extra_valid_dataloader)
        self.model.train()
        report_counter  = 0
        total_counter = 0
        epoch_train_loss = 0.0
        batch_train_loss = 0.0
        batch_loss_counter = 0
        for epoch in range(self.epochs):
            #train
            epoch_train_loss = 0.0
            for step,batch in enumerate(train_dataloader):
                report_counter += len(batch[0])
                total_counter += len(batch[0])
                text, text_mask,labels = batch
                text = text.to(device)
                labels = labels.to(device)    

                self.optimizer.zero_grad()
                logits = self.forward(text,text_mask)
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
                    self.validation(valid_dataloader, replaced_dataloader,embedding_sens_eval_dataloader,epoch,device,False,extra_valid_dataloader)


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


            logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_train_loss/(step+1):.4f}")

