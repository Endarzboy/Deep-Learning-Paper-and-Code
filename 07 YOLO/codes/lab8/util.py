import torch

def non_maximal_suppresion(boxes, probs, threshold=0.5):
    xmin = boxes[:,0]
    ymin = boxes[:,1]
    xmax = boxes[:,2]
    ymax = boxes[:,3]

    areas = (xmax - xmin) * (ymax - ymin)
    
    _, order = probs.sort(0, descending=True)

    remain = []
    
    while order.numel() > 0:

        if order.numel() > 1:
            i = order[0]
        else:
            i = order.item()
            
        remain.append(i)
        
        if order.numel() == 1:
            break
            
        inter_xmin = xmin[order[1:]].clamp(min=xmin[i])
        inter_ymin = ymin[order[1:]].clamp(min=ymin[i])
        inter_xmax = xmax[order[1:]].clamp(max=xmax[i])
        inter_ymax = ymax[order[1:]].clamp(max=ymax[i])
        
        w = (inter_xmax - inter_xmin).clamp(min=0)
        h = (inter_ymax - inter_ymin).clamp(min=0)
        
        inter_area = w * h
        anchor_area = areas[i].expand_as(areas[order[1:]])
        least_area = areas[order[1:]]
        
        iou = inter_area / (anchor_area + least_area - inter_area)
        
        ids = (iou <= threshold).nonzero().squeeze()
        
        
        if ids.numel() == 0:
            break
            
        order = order[ids+1]
        
    return torch.LongTensor(remain)


def interpret_target(pred, threshold):

    pred_boxes = []
    class_ids = []
    pred_probs = []
    
    cell_size = 1./ 7
    pred = pred.data.squeeze(0)
    
    confidence = torch.cat((pred[:,:,4].unsqueeze(2), pred[:,:,9].unsqueeze(2)), 2)
    conf_mask = ((confidence > 0) + (confidence==confidence.max())).gt(0)
    
    for i in range(7):
        for j in range(7):
            for b in range(2):
                if conf_mask[i,j,b] == 1:
                    box = pred[i,j,b*5:b*5+4]
                    conf_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size
                    
                    box[:2] = box[:2]*cell_size + xy
                    box_coord = torch.FloatTensor(box.size())
                    
                    box_coord[:2] = box[:2] - 0.5 * box[2:]
                    box_coord[2:] = box[:2] + 0.5 * box[2:]
                    
                    max_prob, cls_index = torch.max(pred[i,j,10:], 0)
                    
                    if float((conf_prob*max_prob)[0]) > threshold: ## When visualizing the results, it is set for 0.1,
                        pred_boxes.append(box_coord.view(1,4))     ## When evaluating the mAP, it is set for 1e-7
                        class_ids.append(cls_index)
                        pred_probs.append(conf_prob * max_prob)
                        
    if len(pred_boxes) == 0:
        pred_boxes = torch.zeros((1,4))
        pred_probs = torch.zeros(1)
        class_ids = torch.zeros(1)
    else:
        pred_boxes = torch.cat(pred_boxes, 0)
        pred_probs = torch.cat(pred_probs, 0)
        class_ids = torch.stack(class_ids, 0) 
        
    remain = non_maximal_suppresion(pred_boxes, pred_probs)
    
    return pred_boxes[remain], class_ids[remain], pred_probs[remain]
    
