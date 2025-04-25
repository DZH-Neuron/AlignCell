# train.py
import time
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math

# tripletloss
class TripletLoss(nn.Module):   #Triplet Loss
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
 
    def forward(self, anchor, positive, negative):
        pos_dist = F.cosine_similarity(anchor, positive, dim=1)

        neg_dist = F.cosine_similarity(anchor, negative, dim=1)

        loss = torch.relu(neg_dist - pos_dist + self.margin)
        return loss.mean()


def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    criterion = TripletLoss(margin=0.5)
    t_batch = len(train)
    v_batch = len(valid) 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    best_loss = math.inf
    total_loss, totalv_loss = 0, 0
    train=train
    valid=valid
    print("start1\n")
    for epoch in range(n_epoch):
        total_loss = 0
        epoch_start_time = time.time()
        loss_record = []
        print("start2\n")
        
        for i, (anchor_label, anchor_gene, positive_label, positive_gene, negative_label, negative_gene) in enumerate(train):
            start_time = time.time()
            anchor_label, anchor_gene = anchor_label.to(device, dtype=torch.long), anchor_gene.to(device, dtype=torch.long)
            positive_label, positive_gene = positive_label.to(device, dtype=torch.long), positive_gene.to(device, dtype=torch.long)
            negative_label, negative_gene = negative_label.to(device, dtype=torch.long), negative_gene.to(device, dtype=torch.long)
            optimizer.zero_grad() # 由于loss.backward（）的gradient会累加，所以每次喂完一个batch后需要归零
            anchor_out = model(anchor_gene) # 將 input 喂给模型
            positive_out = model(positive_gene)
            negative_out = model(negative_gene)
            
            loss = criterion(anchor=anchor_out,positive=positive_out,negative=negative_out)
            loss.backward() # 算 loss 的 gradient
            optimizer.step()
            loss_record.append(loss.detach().item())
            mean_train_loss = sum(loss_record)/len(loss_record)
            
            print('[ Epoch{}: {}/{} ] Train | loss:{:.5f} '.format(
            	epoch+1, i+1, t_batch, loss.item()), end='\r')
        print('\n[%03d/%03d] %2.2f sec(s)\nTrain | Loss:%3.5f ' % \
              (epoch + 1, n_epoch, time.time()-epoch_start_time, \
                mean_train_loss))
        

        model.eval()
        loss_record = []
        #ret_output = []
        with torch.no_grad():
            totalv_loss = 0
            for i, (anchor_label, anchor_gene, positive_label, positive_gene, negative_label, negative_gene) in enumerate(valid):
                anchor_label, anchor_gene = anchor_label.to(device, dtype=torch.long), anchor_gene.to(device, dtype=torch.long) 
                positive_label, positive_gene = positive_label.to(device, dtype=torch.long), positive_gene.to(device, dtype=torch.long)
                negative_label, negative_gene = negative_label.to(device, dtype=torch.long), negative_gene.to(device, dtype=torch.long)
                anchor_out = model(anchor_gene) # 將 input 喂给模型
                positive_out = model(positive_gene)
                negative_out = model(negative_gene)
                loss = criterion(anchor=anchor_out,positive=positive_out,negative=negative_out) # 计算此时模型的validation loss
                loss_record.append(loss.item())
                mean_valid_loss = sum(loss_record)/len(loss_record)
            #return labels, ret_output   
            print('[ Epoch{}: {}/{} ] Valid | loss:{:.5f} '.format(
            	epoch+1, i+1, t_batch, loss.item()), end='\r')
            print("\nValid | Loss:{:.5f}  ".format(mean_valid_loss))
            #print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(totalv_loss/v_batch, totalv_acc/len(val_dataset)))
            if mean_valid_loss < best_loss:
                best_loss = mean_valid_loss
                # 如果validation的结果优于之前所有的结果，就把当下的模型存下来以备之后做预测时使用
                torch.save(model, "{}/FineTurn_qian_e_mtDNA_new_raw.model".format(model_dir))
                print('saving model with loss {:.5f}   <<========='.format(mean_valid_loss))
                #print('label:',labels.cuda())
                #print('out:',outputs.float())
        print('-----------------------------------------------')
        model.train() # 将model的模式设为train，这样optimizer就可以更新model的参数（因为刚刚转成eval模式）
