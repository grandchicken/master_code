from datetime import datetime
from utils import liner_warmup,set_lr,clip_gradient
from transformers import AdamW
import torch
import fitlog

def finetune(args,model,train_dataloader,dev_dataloader,test_dataloader,metric,):
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    best_dev = 0
    best_model = model
    step = 0
    start_time = datetime.now()
    for epoch in range(args.epochs):
        #记录
        total_step = len(train_dataloader)
        for i,batch in enumerate(train_dataloader):
            step += 1
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            mner_label = batch['mner_label']
            mner_label_masks = batch['mner_label_masks']
            loss = model.forward(input_ids,attention_mask,mner_label,mner_label_masks)
            fitlog.add_loss(loss, name="Loss", step=step)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, args.epochs, i + 1, total_step, loss.item()))
            cur_step = i + 1 + epoch * total_step
            t_step = args.epochs * total_step
            liner_warm_rate = liner_warmup(cur_step, t_step, args.warmup) # 线性warm up 学习率
            set_lr(optimizer, liner_warm_rate * args.lr)
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, args.grad_clip)
            optimizer.step()


        if (epoch+1) % args.eval_every == 0:
            res_dev = eval_utils(args, model, dev_dataloader, metric, device = args.device)
            f1_score = res_dev['aesc_f']

            fitlog.add_metric({"dev": {"f1": f1_score}}, step=step)
            if f1_score > best_dev:
                best_dev = f1_score
                best_model = model
                fitlog.add_best_metric({"dev": {"f1": best_dev}})
            end_time = datetime.now()
            res_dev_str = 'DEV  aesc_p:{} aesc_r:{} aesc_f:{} during time:{}s'.format(
                res_dev['aesc_pre'], res_dev['aesc_rec'], res_dev['aesc_f'], (end_time - start_time).seconds)
            print(res_dev_str)



    res_test = eval_utils(args, best_model, test_dataloader, metric, device=args.device)
    res_test_str = 'TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        res_test['aesc_pre'], res_test['aesc_rec'],
        res_test['aesc_f'])
    print(res_test_str)
    fitlog.add_best_metric({"test": {"f1": res_test['aesc_f']}})
    print('total spend time... : {}s'.format((datetime.now() - start_time).seconds))
    print('save_best_model...')
    torch.save(best_model.state_dict(),'bartner.pth')

# 验证
def eval_utils(args, model, loader, metric, device):
    model.eval()
    for i, batch in enumerate(loader):
        print('{}/{} evaling...'.format(i,len(loader)))
        # Forward pass

        mner_label = batch['mner_label']
        gt_spans = batch['gt_spans']

        predict = model.predict(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            mner_label=mner_label)

        metric.evaluate(gt_spans, predict,
                        mner_label.to(device))
    res = metric.get_metric()
    model.train()
    return res

# test
def direct_test(args,model,metric,test_dataloader):
    device = args.device
    state_dict = torch.load('bartner.pth')
    model.load_state_dict(state_dict)
    model.eval()
    for i, batch in enumerate(test_dataloader):
        mner_label = batch['mner_label']
        gt_spans = batch['gt_spans']
        predict = model.predict(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            mner_label=mner_label)

        metric.evaluate(gt_spans, predict,
						mner_label.to(device))
    res = metric.get_metric()
    return res

# inference
def predict(args, model, test_dataloader):
    # 给出预测的 mner index 值
    model.eval()
    device = args.device
    tot_predict = []
    tot_inputids = []
    total_labels = []
    for i, batch in enumerate(test_dataloader):
        mner_label = batch['mner_label']
        gt_spans = batch['gt_spans']
        predict = model.predict(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            mner_label=mner_label)
        tot_predict.append(predict.cpu())
        total_labels.append(mner_label)
        tot_inputids.append(batch['input_ids'].cpu())
    return tot_predict,tot_inputids,total_labels
