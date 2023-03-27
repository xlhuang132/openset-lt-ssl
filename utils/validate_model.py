import torch
import time
from utils import AverageMeter, FusionMatrix
from tqdm import tqdm
def validate(valloader, model, criterion, cfg=None,mode=None,epoch=None,return_class_acc=False ):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter() 
    # switch to evaluate mode
    model.eval()

    end = time.time() 
    fusion_matrix = FusionMatrix(cfg.DATASET.NUM_CLASSES)
    func = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for  i, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            if len(outputs)==2:
                outputs=outputs[0]
            loss = criterion(outputs, targets)

            # measure accuracy and record loss 
            losses.update(loss.item(), inputs.size(0)) 
            score_result = func(outputs)
            now_result = torch.argmax(score_result, 1) 
            fusion_matrix.update(now_result.cpu().numpy(), targets.cpu().numpy())
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time() 
    group_acc=fusion_matrix.get_group_acc(cfg.DATASET.GROUP_SPLITS)
    acc=fusion_matrix.get_accuracy()
    if return_class_acc: 
        class_acc=fusion_matrix.get_acc_per_class()
        return (losses.avg, acc,group_acc,class_acc)
   
    return (losses.avg, acc, group_acc)
 