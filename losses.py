import torch 
import torch.nn as nn

def triplet_loss(features, margin, batch_size):
    anchor = features[0:batch_size]
    positive = features[batch_size:batch_size*2]
    negative = features[batch_size*2:batch_size*3]
    distance1 = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), 1, keepdims=True))
    distance2 = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), 1, keepdims=True))
    l = distance1-distance2+margin
    for i in range(len(l)):
        if l[i] <0:
            l[i] = 0
    return(torch.mean(l))

def xent_loss(output, trainY, num_classes):
    epsilon = 1.0
    logsoftmax = nn.LogSoftmax(dim=1)
    log_probs = logsoftmax(output)
    targets = torch.zeros(log_probs.size())
    for i in range(len(targets)):
        for j in range(len(targets[i])):
            if j == trainY[i]:
                targets[i][j] = torch.tensor(1.0)
                break
    targets = targets.cuda()
    targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (- targets * log_probs).mean(0).sum()
    return loss