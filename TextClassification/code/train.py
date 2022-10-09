from torch.optim import Adam
import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score,accuracy_score
class Trainer:
    def __init__(self,args,model,label2id,train_dataloader,dev_dataloader,test_dataloader,):
        self.args = args

        self.model = model
        self.label2id = label2id
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.best_dev = 0
        self.best_model = model


    def train(self):
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        for epoch in range(self.args.epochs):
            total_step = len(self.train_dataloader)
            for i,batch in enumerate(self.train_dataloader):
                batch_input_ids,batch_attention_masks,batch_label_ids = batch
                batch_input_ids = batch_input_ids.to(self.args.device)
                batch_attention_masks = batch_attention_masks.to(self.args.device)
                batch_label_ids = batch_label_ids.to(self.args.device)
                output = self.model(batch_input_ids,batch_attention_masks)
                loss = F.cross_entropy(output,batch_label_ids)
                train_info = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, self.args.epochs, i + 1, total_step, loss.item())
                print(train_info)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            eval_acc = self.eval() # 93.22
            dev_info = 'DEV  f1_dev:{}'.format(eval_acc)
            print('----------------------------------')
            print('----------------------------------')
            print(dev_info)
            print('----------------------------------')
            print('----------------------------------')
            if eval_acc > self.best_dev:
                self.best_dev = eval_acc
                self.best_model = self.model
        test_acc = self.test()
        test_info = 'TEST  f1_test:{}'.format(test_acc)
        print('----------------------------------')
        print('----------------------------------')
        print(test_info)
        print('----------------------------------')
        print('----------------------------------')



    def eval(self):
        self.model.eval()
        total_pred = []
        total_true = []
        for i, batch in enumerate(self.dev_dataloader):
            batch_input_ids, batch_attention_masks, batch_label_ids = batch
            batch_input_ids = batch_input_ids.to(self.args.device)
            batch_attention_masks = batch_attention_masks.to(self.args.device)
            batch_label_ids = batch_label_ids.to(self.args.device)
            output = self.model(batch_input_ids, batch_attention_masks)
            pred_logits = torch.argmax(output,dim = -1)
            total_pred.extend(pred_logits.to('cpu').numpy().tolist())
            total_true.extend(batch_label_ids.to('cpu').numpy().tolist())
        acc = accuracy_score(total_true,total_pred)
        return acc

    def test(self):
        self.best_model.eval()
        total_pred = []
        total_true = []
        for i, batch in enumerate(self.test_dataloader):
            batch_input_ids, batch_attention_masks, batch_label_ids = batch
            batch_input_ids = batch_input_ids.to(self.args.device)
            batch_attention_masks = batch_attention_masks.to(self.args.device)
            batch_label_ids = batch_label_ids.to(self.args.device)
            output = self.best_model(batch_input_ids, batch_attention_masks)
            pred_logits = torch.argmax(output,dim = -1)
            total_pred.extend(pred_logits.to('cpu').numpy().tolist())
            total_true.extend(batch_label_ids.to('cpu').numpy().tolist())
        acc = accuracy_score(total_true,total_pred)
        return acc

