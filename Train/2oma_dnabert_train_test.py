import torch
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np
import shutil
import sys
import os

from Bio import SeqIO
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df_x, df_y, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.df_x = df_x
        self.df_y = df_y
        self.max_len = max_len

    def __len__(self):
        return len(self.df_x) ## of sequences: 243 for homo_lncRNA_multi6_seq

    def __getitem__(self, index):
        df_x = str(self.df_x[index])
        df_x = "".join(df_x.split())

        inputs = self.tokenizer(#.encode_plus
            df_x,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'features': inputs["input_ids"].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'labels': torch.FloatTensor(self.df_y[index])
        }


class BERTClass(torch.nn.Module):
    def __init__(self,dp):
        super(BERTClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        #BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.batch_norm = torch.nn.BatchNorm1d(768)
        self.dropout1 = torch.nn.Dropout(dp)
        self.linear1 = torch.nn.Linear(768, 256)
        self.linear2 = torch.nn.Linear(256, 1)
        #self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input_ids, attn_mask, token_type_ids):

        hidden_states = torch.stack(list(self.bert_model(input_ids,
            attention_mask = attn_mask,
            token_type_ids = token_type_ids)[0]),dim=0)
        mean_embeddings = torch.mean(hidden_states, dim=1)
        output_dropout1 = self.dropout1(mean_embeddings)
        batch_norm_out = self.batch_norm(output_dropout1)
        linear_output1 = self.linear1(batch_norm_out)
        linear_output2 = self.linear2(linear_output1)
        
        #output_probabilities = self.sigmoid(linear_output1)
        #del input_ids, mean_embeddings,linear_output1
        torch.cuda.empty_cache()

        return linear_output2

class EarlyStopping:
    def __init__(self, tolerance=3, min_delta=0.4):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def getXY(filename):

    sequences_test = SeqIO.parse(filename, "fasta")

    X, y = [], []
    for record in sequences_test:
        output = ''.join(record.seq)
        X.append(output)
        y.append(record.id[:1])

    X, y = np.array(X), np.array(y)

    return X, y


def getmaxtokenizerlen(X):
    maxlen = 0
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    for each in X:
        templen=tokenizer(each,return_tensors='pt')["input_ids"].shape[1]
        if templen > maxlen:
            maxlen=templen
    print('Max length of tokenizer:')
    print(maxlen)
    return maxlen

def get_loader(X, y, tokenlen,BATCH_SIZE):
    # Convert data to tensors and create DataLoader
    inty = [int(x) for x in y]
    y = torch.tensor(inty)
    dataset_new = CustomDataset(X, y.float(), tokenlen)
    loader_new = torch.utils.data.DataLoader(
        dataset_new,batch_size=BATCH_SIZE,
        shuffle=False,num_workers=0) #shuffle=False 顺序抽样
    return loader_new


def train(X, y, k, tokenlen, LEARNING_RATE, EPOCHS, BATCH_SIZE, dp, MODEL_SAVE_PATH,schedulerstep):
    foldperf = foldoutput = foldtarget = foldauc = {}
    fold = 0

    k_fold = StratifiedKFold(n_splits=k)
    
    for train_index, val_index in k_fold.split(X, y):

        fold += 1

        foldtrain_X, foldtrain_y = X[train_index], y[train_index]
        foldtrain_data_loader = get_loader(foldtrain_X, foldtrain_y, tokenlen, BATCH_SIZE)
        foldval_X, foldval_y = X[val_index], y[val_index]
        foldval_data_loader = get_loader(foldval_X, foldval_y, tokenlen, BATCH_SIZE)

        min_test_loss = 10
        min_test_target = min_test_output = []

        model = BERTClass(dp)
        model.to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
        if schedulerstep:
            scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5, threshold=0.0001, min_lr = 1e-6)

        history = {'test_loss': [], 'test_acc':[], 'test_macro': [], 'test_micro': []}

        early_stopping = EarlyStopping(tolerance=5, min_delta=0.5)

        for epoch in range(1, EPOCHS+1):
            train_loss = test_loss = test_correct = 0

            torch.cuda.empty_cache()
            

            model.train()
            for batch_idx, data in enumerate(foldtrain_data_loader):
                features = data['features'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                labels = data['labels'].to(device, dtype=torch.float).unsqueeze(1)

                outputs = model(features, mask, token_type_ids)
                output_prob = torch.sigmoid(outputs)
                #print(outputs)
                #print(labels)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            train_loss = train_loss/len(foldtrain_data_loader)

            model.eval()
            with torch.no_grad():
                test_targets = []
                test_outputs = []
                for batch_idx, data in enumerate(foldval_data_loader):
                    features = data['features'].to(device, dtype=torch.long)
                    mask = data['attention_mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    labels = data['labels'].to(device, dtype=torch.float).unsqueeze(1)

                    outputs = model(features, mask, token_type_ids)
                    output_prob = torch.sigmoid(outputs)
                    #print(outputs)
                    loss = loss_fn(outputs, labels)
                    test_loss = test_loss + loss.item()
                    test_targets.extend(labels.cpu().detach().numpy().tolist())
                    test_outputs.extend(output_prob.cpu().detach().numpy().tolist())
                # Calculate average losses and accuracies
                test_loss = test_loss / len(foldval_data_loader)

                # Calculate ROC scores
                test_targets = np.array(test_targets)
                test_outputs = np.array(test_outputs)
                # print('*********test_outputs writing*********')
                # writetofile(test_targets, test_outputs)

                predicted_classes = np.round(test_outputs)
                test_accuracy = np.mean(predicted_classes == test_targets)
                
                test_roc_macro = roc_auc_score(test_targets, test_outputs, average='macro')
                #test_roc_micro = roc_auc_score(test_targets, test_outputs, average='micro')

                # Print training/validation statistics
                print("Fold:{}/{}  Epoch:{}/{} Train Loss:{:.4f} Test Loss:{:.4f} Accuracy:{:.4f} ROC Macro: {:.4f} LR: {}"
                    .format(fold,k, epoch, EPOCHS, train_loss, test_loss, test_accuracy, test_roc_macro, optimizer.param_groups[0]['lr']))

                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_accuracy)
                history['test_macro'].append(test_roc_macro)
                #history['test_micro'].append(test_roc_micro)

                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    min_test_target = test_targets
                    min_test_output = test_outputs
                    min_test_auc = test_roc_macro
            early_stopping(train_loss, test_loss)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
        #foldtarget['fold{}'.format(fold)] = min_test_target 
        #foldoutput['fold{}'.format(fold)] = min_test_output
        foldauc['fold{}'.format(fold)] = min_test_auc

        #foldperf['fold{}'.format(fold)] = history 
        
    #roc_micro_mean = np.mean([history['test_micro'][-1] for history in foldperf.values()])
    #test_loss, test_accuracy, test_macro = [], [], []
    print('**************************K fold Train & Evaluation finished, Model saving***************************')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    test_auc = []
    for f in range(1,k+1):
        test_auc.append(foldauc['fold{}'.format(f)])
        #test_loss.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))
        #test_accuracy.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))
        #test_macro.append(np.mean(foldperf['fold{}'.format(f)]['test_macro']))
        #test_micro.append(np.mean(foldperf['fold{}'.format(f)]['test_micro']))
    #print('Performance of {} fold cross validation'.format(k))
    #print("Average Loss: {:.4f} \t Accuracy: {:.4f} \t Macro: {:.4f} \t Micro: {:.4f}".format(np.mean(test_loss),np.mean(test_accuracy),np.mean(test_macro),np.mean(test_micro)))
    fold_auc = np.mean(test_auc)

    return fold_auc

def getperformance(test_targets, test_outputs):

    # Calculate ROC scores on the new test set
    test_targets = np.array(test_targets)
    test_outputs = np.array(test_outputs)
    test_roc_macro = roc_auc_score(test_targets, test_outputs, average='macro')
    #test_roc_micro = roc_auc_score(test_targets, test_outputs, average='micro')

    # Print evaluation metrics on the new test set
    #print("Evaluation on the new test set:")
    #print("ROC Macro: {:.4f} ROC Micro: {:.4f}".format(test_roc_macro, test_roc_micro))
    return test_roc_macro

def writetofile(test_targets, test_outputs):
    gt=open('target.csv','w')
    go=open('output.csv','w')

    for eachl in test_targets:
        templine=''
        for eacht in eachl:
            templine += ',%s'%eacht
        gt.write('%s\n'%templine.strip(','))
    gt.close()

    for eachl in test_outputs:
        templine=''
        for eacht in eachl:
            templine += ',%s'%eacht
        go.write('%s\n'%templine.strip(','))
    go.close()

def evaluation(X, y, tokenlen, BATCH_SIZE,bp,MODEL_SAVE_PATH,wf=False):
    test_loss = 0
    test_targets = []
    test_outputs = []

    test_loader = get_loader(X, y, tokenlen, BATCH_SIZE)

    model = BERTClass(bp)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            features = data['features'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float).unsqueeze(1)

            outputs = model(features, mask, token_type_ids)
            loss = loss_fn(outputs, labels)
            test_loss = test_loss + loss.item()#test_loss + ((1 / (batch_idx + 1)) * loss.item())

            test_targets.extend(labels.cpu().detach().numpy().tolist())
            test_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    test_loss = test_loss / len(test_loader)
    #print("test loss:%s"%test_loss)

    test_roc_macro = getperformance(test_targets, test_outputs)

    if wf:
        writetofile(test_targets, test_outputs)
        print("Targets and Outputs of test dataset have been write to files.")
    return test_roc_macro

#
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# hyperparameters

BATCH_SIZE = 512
EPOCHS = 100
#LEARNING_RATE = 3e-2
NUMS_LABELS = 1
OUTPUT_SIZE = 1
k = 10 #fold
#bp = 0.3

#train
train_filename = "2omA.fasta"
X, y = getXY(train_filename)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify= y)

tokenlen = 10#getmaxtokenizerlen(X_train)# A 10 C 10 G 15
g=open('2oma_res.csv','a')
g.write("Drop_out,schedulerstep,LEARNING_RATE,CrossValidationAUC,TestAUC\n")
MODEL_SAVE_PATH = "dnabert_model.pt"
for dp in [0,0.01,0.1,0.2,0.3,0.4,0.5]:
    for schedulerstep in [True,False]:
        for LEARNING_RATE in [1e-5,3e-5,5e-5]:
            train_roc_macro=train(X_train, y_train, k, tokenlen, LEARNING_RATE, EPOCHS, BATCH_SIZE, dp, MODEL_SAVE_PATH,schedulerstep)
            # Load the homo test set
            test_roc_macro=evaluation(X_test, y_test, tokenlen, BATCH_SIZE, dp, MODEL_SAVE_PATH,wf=False)
            g.write("%s,%s,%s,%s,%s\n"%(dp,schedulerstep,LEARNING_RATE,train_roc_macro,test_roc_macro))
g.close()
