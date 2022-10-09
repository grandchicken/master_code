from torch.optim import Adam
from tqdm import tqdm
from seqeval.metrics import classification_report,f1_score
import fitlog


class Trainer:
    def __init__(self,args,model,label2ids,train_dataloader,dev_dataloader,test_dataloader):
        self.args = args
        self.label2ids = label2ids
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.best_model = model
        self.best_dev = 0
        self.best_loss = 10


    def train(self):
        step = 0
        optimizer = Adam(self.model.parameters(),lr = self.args.lr)
        for epoch in range(self.args.epochs):
            epoch_step = len(self.train_dataloader)
            for i,batch in enumerate(self.train_dataloader):
                step += 1
                input_id, attention_mask, segment_id, label_id, image_feat, image_mask = batch
                input_id, attention_mask, segment_id, label_id, image_feat, image_mask = \
                    input_id.cuda(), attention_mask.cuda(), segment_id.cuda(), label_id.cuda(), image_feat.cuda(), image_mask.cuda()
                loss = self.model(input_id, attention_mask, segment_id, label_id, image_feat, image_mask)
                fitlog.add_loss(loss,name='Loss',step = step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % self.args.every_step == 0:
                    if loss < self.best_loss:
                        self.best_loss = loss
                        mark = '*'
                    else:
                        mark = ''

                    train_info = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} '.format(
                        epoch + 1, self.args.epochs, i + 1, epoch_step, loss.item()) + mark
                    print(train_info)

            eval_f1 = self.eval()
            dev_info = 'DEV  f1_dev:{}'.format(eval_f1)
            # wandb.log({'eval_f1': eval_f1})
            fitlog.add_metric({'dev':{'f1':eval_f1}},step = step)
            print(dev_info)
            if eval_f1 > self.best_dev:
                self.best_dev = eval_f1
                self.best_model = self.model
                # fitlog.add_best_metric({'dev': {'f1': eval_f1}})

        test_f1,c_report = self.test()
        # wandb.log({'test_f1': test_f1})
        test_info = 'TEST  f1_dev:{}'.format(test_f1)
        fitlog.add_best_metric({'test': {'f1': test_f1}})
        print(test_info)
        print(c_report)

    def eval(self):
        self.model.eval()
        total_y_pred = []
        total_y_true = []
        for i,batch in tqdm(enumerate(self.dev_dataloader),desc='Evaluating...'):
            input_id, attention_mask, segment_id, label_id, image_feat, image_mask= batch
            input_id, attention_mask, segment_id, label_id, image_feat, image_mask = \
                input_id.cuda(), attention_mask.cuda(), segment_id.cuda(), label_id.cuda(), image_feat.cuda(), image_mask.cuda()
            pred_logits = self.model(input_id, attention_mask, segment_id, label_id, image_feat, image_mask, is_training = False)
            label_id = label_id.to('cpu').numpy()
            y_pred, y_true = self.transform_NER(pred_logits,label_id,attention_mask)
            total_y_true.extend(y_true)
            total_y_pred.extend(y_pred)
        f1 = f1_score(total_y_true,total_y_pred)
        self.model.train()
        return f1

    def test(self):
        self.best_model.eval()
        total_y_pred = []
        total_y_true = []
        for i,batch in tqdm(enumerate(self.test_dataloader),desc='Test...'):
            input_id, attention_mask, segment_id, label_id, image_feat, image_mask = batch
            input_id, attention_mask, segment_id, label_id, image_feat, image_mask = \
                input_id.cuda(), attention_mask.cuda(), segment_id.cuda(), label_id.cuda(), image_feat.cuda(), image_mask.cuda()
            pred_logits = self.best_model(input_id, attention_mask, segment_id, label_id, image_feat, image_mask, is_training = False)
            label_id = label_id.to('cpu').numpy()
            y_pred, y_true = self.transform_NER(pred_logits,label_id,attention_mask)
            total_y_true.extend(y_true)
            total_y_pred.extend(y_pred)
        f1 = f1_score(total_y_true,total_y_pred)
        c_report = classification_report(total_y_true,total_y_pred)
        self.best_model.train()
        return f1,c_report

    def transform_NER(self,pred_logits,label_ids,attention_mask):
            reverse_label_map = {value:key for key,value in self.label2ids.items()}
            y_pred = []
            y_true = []
            for i, mask in enumerate(attention_mask):
                temp_true = []
                temp_pred = []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if reverse_label_map[label_ids[i][j]] != "X":
                            temp_true.append(reverse_label_map[label_ids[i][j]])
                            temp_pred.append(reverse_label_map[pred_logits[i][j]])
                    else:
                        break
                y_pred.append(temp_pred)
                y_true.append(temp_true)
            return y_pred,y_true