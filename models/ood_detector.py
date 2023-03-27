import torch
import numpy as np
from tqdm import tqdm
import os

def count_result(targets,is_correct,num_classes=10):
    # (ID分为ID + OOD 分为OOD)/total 
    """
    inputs: 
        tarrget: class_label[0,num_classes) -1:target
        is_correct: 是否检测正确- ID样本为1 OOD样本为0
    outputs:  
        total_ood_correct: OOD有多少被正确分为OOD
        total_ood_wrong: 
        class_correct:每类有多少被正确分为ID
        class_wrong:
    """ 
    total_ood_correct,total_ood_wrong=0,0
    class_correct,class_wrong=np.array([0]*num_classes),np.array([0]*num_classes)
    
    for i in range(len(is_correct)):
        if is_correct[i]: 
            if targets[i]==-1:
                total_ood_correct+=1
            else:
                class_correct[targets[i]]+=1
        else: 
            if targets[i]==-1:
                total_ood_wrong+=1
            else:
                class_wrong[targets[i]]+=1 
    
    return class_correct,class_wrong,total_ood_correct,total_ood_wrong

def perturb_input(model, images, epsilon, temperature):
    """
    Execute adversarial attack on the input image.
    :param model: pytorch model to use.
    :param images: image to attack.
    :param epsilon: the attack strength
    :param temperature: smoothing factor of the logits.
    :param model_name: name of architecture
    :return: attacked image
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad()

    # Forward
    images.requires_grad = True
    outputs = model(images)

    # Using temperature scaling
    outputs = outputs / temperature

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    pseudo_labels = torch.argmax(outputs, dim=1).detach()
    loss = criterion(outputs, pseudo_labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(images.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    # resnet
    gradient.index_copy_(
        1,
        torch.LongTensor([0]).cuda(),
        gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023),
    )
    gradient.index_copy_(
        1,
        torch.LongTensor([1]).cuda(),
        gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994),
    )
    gradient.index_copy_(
        1,
        torch.LongTensor([2]).cuda(),
        gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010),
    )
    # Adding small perturbations to images
    image_perturbs = torch.add(images.data, -gradient, alpha=epsilon)
    image_perturbs.requires_grad = False
    model.zero_grad()
    return image_perturbs

def get_fpr95(file_path,name):
    assert os.path.exists(file_path) 
    data = np.loadtxt(file_path, delimiter=',')
    T = 1000 
    if name == "cifar10": 
        start = 0.1
        end = 0.12 
        num_classes=10
    if name == "cifar100": 
        start = 0.01
        end = 0.0104        
        num_classes=100
    gap = (end- start)/100000  
    total = 0.0
    fpr = 0.0
    targets,domain_labels,confidence=data[:,0],data[:,1],data[:,2]
    # class_tprs=[]
    # id_tprs,ood_tprs,total_tprs=[],[],[]
    thresh=np.arange(start, end, gap)
    targets=torch.Tensor(targets).cuda()
    domain_labels=torch.Tensor(domain_labels).cuda()
    confidence=torch.Tensor(confidence).cuda()    
    zeros = torch.zeros_like(targets).cuda()
    ones=torch.ones_like(targets).cuda() 
    for delta in tqdm(thresh):
        class_correct,class_wrong=np.array([0]*num_classes),np.array([0]*num_classes)
        for i in range(num_classes):
            class_correct[i]=( torch.where(confidence >= delta,ones,zeros)* torch.where(domain_labels==1,ones,zeros)   * torch.where(targets==i,ones,zeros)).sum()  
            class_wrong[i]=( torch.where(confidence < delta,ones,zeros)* torch.where(domain_labels==1,ones,zeros)   * torch.where(targets==i,ones,zeros)).sum()  
        ood_correct=(torch.where(confidence < delta,ones,zeros)* torch.where(domain_labels==0,ones,zeros)).sum()
        ood_wrong=(torch.where(confidence >= delta,ones,zeros) *torch.where(domain_labels==0,ones,zeros)).sum()
        # 正确分类/total
        total_id=sum(class_correct)+sum(class_wrong)
        total_ood=ood_correct+ood_wrong 
        total_num=(total_id+total_ood) 
        # 每个类的tpr
        # class_tpr=class_correct/(class_correct+class_wrong)
        # class_tprs.append(class_tpr)
        # id_tprs.append(sum(class_correct)/total_id)
        # ood_tprs.append((ood_correct/total_ood).item())
        tpr =  (sum(class_correct)+ood_correct)/total_num
        
        
        # total_tprs.append(tpr.item())
        error2 = (sum(class_wrong)+ood_wrong)/total_num
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
            
    fpr95 = fpr/total 
    return fpr95
    # class_tpr= 
    # return thresh,class_tprs,id_tprs,ood_tprs,total_tprs
# ,fpr95
    
    
def ood_detect_batch(model,inputs,cfg):
    
    eps=cfg.ALGORITHM.OOD_DETECTOR.MAGNITUDE
    temperature=cfg.ALGORITHM.OOD_DETECTOR.TEMPERATURE
    logits = model(inputs) 
    # Fine-tune image
    if eps > 0.0:
        with torch.enable_grad():
            x_perturbs = perturb_input(
                model, inputs, eps,temperature
            )
    else:
        x_perturbs = x
    # ====  forward  =====
    with torch.no_grad():
            logits,features = model(x_perturbs,return_encoding=True) 
            norm = torch.linalg.norm(features, dim=-1, keepdim=True)
            features = features / norm

            # Forward with feature normalization
            logits_w_norm_features = model(features,classifier=True) 
    
    logits = logits / temperature
    # logits_w_norm_features = logits_w_norm_features /temperature
    p = torch.nn.functional.softmax(logits, dim=-1)
    # probs_normalized = torch.nn.functional.softmax(logits_w_norm_features, dim=-1) 
    confidence, _ = torch.max(p, dim=1)
    id_mask =confidence.ge(cfg.ALGORITHM.CONFIDENCE_THRESHOLD ).float() 
    return id_mask
    
def ood_detect_ODIN(model,domain_loader,cfg,confidence_save_path):
    # model.train() 
    eps=cfg.ALGORITHM.OOD_DETECTOR.MAGNITUDE
    temperature=cfg.ALGORITHM.OOD_DETECTOR.TEMPERATURE
    num_classes=cfg.DATASET.NUM_CLASSES
    class_correct=np.array([0]*num_classes)
    class_wrong=np.array([0]*num_classes)
    total_ood_correct,total_ood_wrong=0,0
    dataset=cfg.DATASET.NAME
    # confidence_save_path="./softmax_scores/confidence_Our_In.txt"
    g1 = open(confidence_save_path, 'w')
    for i, (inputs, targets, domain_labels, indexs) in tqdm(enumerate(domain_loader)):

        inputs, domain_labels = inputs.cuda(), domain_labels.cuda(non_blocking=True)

        logits = model(inputs) 
    
        # Fine-tune image
        if eps > 0.0:
            with torch.enable_grad():
                x_perturbs = perturb_input(
                    model, inputs, eps,temperature
                )
        else:
            x_perturbs = x
        
        # ====  forward  =====
        with torch.no_grad():
                logits,features = model(x_perturbs,return_encoding=True) 
                norm = torch.linalg.norm(features, dim=-1, keepdim=True)
                features = features / norm

                # Forward with feature normalization
                logits_w_norm_features = model(features,classifier=True) 
        
        logits = logits / temperature
        # logits_w_norm_features = logits_w_norm_features /temperature
        # p = torch.nn.functional.softmax(logits, dim=-1)
        # probs_normalized = torch.nn.functional.softmax(logits_w_norm_features, dim=-1) 
        # confidence, _ = torch.max(p, dim=1)
        
        logits = logits.data.cpu()
        logits = logits.numpy()
        logits = logits[0]
        logits = logits - np.max(logits)
        logits = np.exp(logits)/np.sum(np.exp(logits))
          
        # confidence>thresh就是ID
        # conf_thr=1/cfg.DATASET.NUM_CLASSES #  为什么使用分类数量的倒数作为阈值呢？    
        confidence= np.max(logits)
        g1.write("{}, {}, {}\n".format(targets[0].item(),domain_labels[0].item(),confidence))
        # confidence, y_hat= torch.max(logits, dim=1) 
        # 是否是ID
         
        # 
        # c_correct,c_wrong,ood_correct,ood_wrong=count_result( targets,is_correct = id_mask == domain_labels,num_classes=cfg.DATASET.NUM_CLASSES)       
        # class_correct+=c_correct
        # class_wrong+=c_wrong
        # total_ood_correct+=ood_correct
        # total_ood_wrong+=ood_wrong
    g1.close()
    # 计算每个类的TPR=(tp)/(tp+fn) # fpr95
    return threshes,class_tprs,id_tprs,ood_tprs,total_tprs 
    # class_tpr=class_correct/(class_correct+class_wrong)
    # id_tpr=class_correct.sum()/(class_correct.sum()+class_wrong.sum())
    # ood_tpr=total_ood_correct/(total_ood_correct+total_ood_wrong)
    # total_tpr=(class_correct.sum()+total_ood_correct)/(total_ood_correct+total_ood_wrong+class_correct.sum()+class_wrong.sum())
    
    # return  class_tpr,id_tpr,ood_tpr,total_tpr

def ood_test(model, criterion,  id_test_loader, ood_testloader, nnName, dataName, magnitude, temper):
    t0 = time.time()  
    g1 = open("./softmax_scores/confidence_ODIN_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_ODIN_Out.txt", 'w')
    N = 10000 
    print("Processing in-distribution images")
    model.eval()
########################################In-distribution###########################################
    for j, data in enumerate(id_test_loader):
        if j<1000: continue
        images, _ = data
        
        inputs =  images.cuda()
        outputs = model(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        logits = outputs.data.cpu()
        logits = logits.numpy()
        logits = logits[0]
        logits = logits - np.max(logits)
        logits = np.exp(logits)/np.sum(np.exp(logits))
        
        # Using temperature scaling
        outputs = outputs / temper
	
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(logits)
        labels = torch.LongTensor([maxIndexTemp]).cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -magnitude, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        logits = outputs.data.cpu()
        logits = logits.numpy()
        logits = logits[0]
        logits = logits - np.max(logits)
        logits = np.exp(logits)/np.sum(np.exp(logits)) 
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()
        
        if j == N - 1: break


    t0 = time.time()
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    for j, data in enumerate(ood_testloader):
        if j<1000: continue
        images, _ = data
    
    
        inputs = images.cuda() 
        outputs = model(inputs)
        


        # Calculating the confidence of the output, no perturbation added here
        logits = outputs.data.cpu()
        logits = logits.numpy()
        logits = logits[0]
        logits = logits - np.max(logits)
        logits = np.exp(logits)/np.sum(np.exp(logits))
        f2.write("{}, {}, {}\n".format(temper, magnitude, np.max(logits)))
        
        # Using temperature scaling
        outputs = outputs / temper
  
  
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(logits)
        labels = torch.LongTensor([maxIndexTemp]).cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -magnitude, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        logits = outputs.data.cpu()
        logits = logits.numpy()
        logits = logits[0]
        logits = logits - np.max(logits)
        logits = np.exp(logits)/np.sum(np.exp(logits))
        g2.write("{}, {}, {}\n".format(temper, magnitude, np.max(logits)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        if j== N-1: break


