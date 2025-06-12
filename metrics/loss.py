import torch
import torch.nn as nn

class PatchLoss(nn.Module):
    def __init__(self, config):
        super(PatchLoss, self).__init__()
        self.config = config
        self.device = config.experiment.device
        self.ignore_label = config.train.ignore_label


    def compute_gradloss(self, model_output, label):
        """
        Compute the adaptive loss function
        """

        #print(model_output.shape,true_labels.shape)
        #print(model_output.argmax(dim=1).shape)
        ce_loss = nn.CrossEntropyLoss(reduction="none",
                                      ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss_map = ce_loss(model_output, label.long())  # Compute loss for all pixels
        #print(f'loss map: {loss_map.shape}')
              
      
        # Get correctly classified and misclassified pixel sets
        predict = torch.argmax(model_output, 1).float() + 1
        target = label.float() + 1
        target[target>=255] = 0
        # print(predict.dtype,predict.shape,target.dtype,target.shape)
        # temp1 = (predict == target).float()
        # temp2 = (target>0).float()
        # print(temp1.dtype,temp2.dtype)
        correct_mask = (predict == target)*(target > 0)
        incorrect_mask = (predict != target)*(target > 0)  # Opposite of correctly classified
        #print(f'Correct mask: {correct_mask.shape}')  
        # Compute separate loss terms
        loss_correct = (loss_map * correct_mask).sum()/correct_mask.sum()  # Loss on correctly classified pixels
        loss_incorrect = (loss_map * incorrect_mask).sum()/incorrect_mask.sum()  # Loss on already misclassified pixels

        # Compute adaptive balancing factor
        num_correct = correct_mask.sum()
        num_total = (target != 0).sum()
        gamma = num_correct / num_total  # Avoid division by zero

        # Final adaptive loss
        loss = gamma * loss_correct + (1 - gamma) * loss_incorrect
        # print(f'Gamma:{gamma}')
        # print(f'loss correct:{loss_correct}')
        # print(f'loss incorrect: {loss_incorrect}')
        #return loss
        return loss_correct 

def compute_trainloss(self, model_output, label):
        """
        Compute the adaptive loss function
        """

        #print(model_output.shape,true_labels.shape)
        #print(model_output.argmax(dim=1).shape)
        ce_loss = nn.CrossEntropyLoss(reduction="none",
                                      ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss_map = ce_loss(model_output, label.long())  # Compute loss for all pixels
        #print(f'loss map: {loss_map.shape}')
              
      
        # Get correctly classified and misclassified pixel sets
        predict = torch.argmax(model_output, 1).float() + 1
        target = label.float() + 1
        target[target>=255] = 0
        # print(predict.dtype,predict.shape,target.dtype,target.shape)
        # temp1 = (predict == target).float()
        # temp2 = (target>0).float()
        # print(temp1.dtype,temp2.dtype)
        correct_mask = (predict == target)*(target > 0)
        incorrect_mask = (predict != target)*(target > 0)  # Opposite of correctly classified
        #print(f'Correct mask: {correct_mask.shape}')  
        # Compute separate loss terms
        loss_correct = (loss_map * correct_mask).sum()/correct_mask.sum()  # Loss on correctly classified pixels
        loss_incorrect = (loss_map * incorrect_mask).sum()/incorrect_mask.sum()  # Loss on already misclassified pixels

        # Compute adaptive balancing factor
        num_correct = correct_mask.sum()
        num_total = (target != 0).sum()
        gamma = num_correct / num_total  # Avoid division by zero

        # Final adaptive loss
        loss = gamma * loss_correct + (1 - gamma) * loss_incorrect
        # print(f'Gamma:{gamma}')
        # print(f'loss correct:{loss_correct}')
        # print(f'loss incorrect: {loss_incorrect}')
        #return loss
        return loss_correct

    def compute_loss_direct(self, model_output, label):
        """
        Compute the adaptive loss function
        """
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss = ce_loss(model_output, label.long())  # Compute loss for all pixels
        return loss 
