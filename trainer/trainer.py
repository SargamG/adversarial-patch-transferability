import sys
sys.path.append("/kaggle/working/adversarial-patch-transferability")
from dataset.cityscapes import Cityscapes

from pretrained_models.models import Models

from pretrained_models.ICNet.icnet import ICNet
from pretrained_models.BisNetV1.model import BiSeNetV1
from pretrained_models.BisNetV2.model import BiSeNetV2
from pretrained_models.PIDNet.model import PIDNet, get_pred_model

from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss
from patch.create import Patch
from torch.optim.lr_scheduler import ExponentialLR
import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

class PatchTrainer():
  def __init__(self,config,main_logger,model_name):
      self.config = config
      self.start_epoch = config.train.start_epoch
      self.end_epoch = config.train.end_epoch
      self.epochs = self.end_epoch - self.start_epoch
      self.batch_train = config.train.batch_size
      self.batch_test = config.test.batch_size
      self.device = config.experiment.device
      self.logger = main_logger
      self.lr = config.optimizer.init_lr
      self.power = config.train.power
      self.lr_scheduler = config.optimizer.exponentiallr
      self.lr_scheduler_gamma = config.optimizer.exponentiallr_gamma
      self.log_per_iters = config.train.log_per_iters
      self.patch_size = config.patch.size
      self.apply_patch = Patch(config).apply_patch
      self.epsilon = config.optimizer.init_lr

      cityscape_train = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.train,
          num_classes = config.dataset.num_classes,
          multi_scale = config.train.multi_scale,
          flip = config.train.flip,
          ignore_label = config.train.ignore_label,
          base_size = config.train.base_size,
          crop_size = (config.train.height,config.train.width),
          scale_factor = config.train.scale_factor
        )

      cityscape_test = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.val,
          num_classes = config.dataset.num_classes,
          multi_scale = False,
          flip = False,
          ignore_label = config.train.ignore_label,
          base_size = config.test.base_size,
          crop_size = (config.test.height,config.test.width),
        )
      
      self.train_dataloader = torch.utils.data.DataLoader(dataset=cityscape_train,
                                              batch_size=self.batch_train,
                                              shuffle=config.train.shuffle,
                                              num_workers=config.train.num_workers,
                                              pin_memory=config.train.pin_memory,
                                              drop_last=config.train.drop_last)
      self.test_dataloader = torch.utils.data.DataLoader(dataset=cityscape_test,
                                            batch_size=self.batch_test,
                                            shuffle=False,
                                            num_workers=config.test.num_workers,
                                            pin_memory=config.test.pin_memory,
                                            drop_last=config.test.drop_last)
      


      self.iters_per_epoch = len(self.train_dataloader)
      self.max_iters = self.end_epoch * self.iters_per_epoch

      ## Getting the model
      self.model1 = Models(self.config)
      self.model1.get()

      self.model2 = Models(self.config)
      self.model2.get()

      ## loss
      self.criterion = PatchLoss(self.config)
      ## optimizer
      # Initialize adversarial patch (random noise)
      self.adv_patch = torch.rand((3, self.patch_size, self.patch_size), 
                              requires_grad=True, 
                              device=self.device)
      self.rand_patch = torch.rand((3, self.patch_size, self.patch_size), 
                              requires_grad=True, 
                              device=self.device)
      
      # # Define optimizer
      # self.optimizer = torch.optim.SGD(params = [self.patch],
      #                         lr=self.lr,
      #                         momentum=config.optimizer.momentum,
      #                         weight_decay=config.optimizer.weight_decay,
      #                         nesterov=config.optimizer.nesterov,
      # )
      # if self.lr_scheduler:
      #   self.scheduler = ExponentialLR(self.optimizer, gamma=self.lr_scheduler_gamma)


      ## Initializing quantities
      self.metric = SegmentationMetric(config) 
      self.current_mIoU = 0.0
      self.best_mIoU = 0.0

      self.current_epoch = 0
      self.current_iteration = 0
      self.model_name=model_name

      # Register hook
      if 'pidnet_s' in self.model_name:
        self.layer_name = 'layer3.2.bn2'  # Change this to the correct intermediate layer
        self.feature_map_shape = [128,64,64]
      elif 'pidnet_m' in self.model_name:
        self.layer_name = 'layer3.2.bn2'
        self.feature_map_shape = [256,64,64]
      elif 'bisenet' in self.model_name:
        self.layer_name = 'segment.S3.1.relu'
        self.feature_map_shape=[32,128,128]
      else:
        self.layer_name = 'pretrained.layer2.3.relu'
        self.feature_map_shape=[512,32,32]

      self.feature_maps_adv = None
      self.feature_maps_rand = None
    
  # Hook to store feature map
  def hook1(self, module, input, output):
      self.feature_maps_adv = output
      output.retain_grad()

  def hook2(self, module, input, output):
      self.feature_maps_rand = output
      output.retain_grad()

  def register_forward_hook1(self):
    for name, module in self.model1.model.named_modules():
        if name == self.layer_name:
          module.register_forward_hook(self.hook1)
          break

  def register_forward_hook2(self):
    for name, module in self.model2.model.named_modules():
        if name == self.layer_name:
          module.register_forward_hook(self.hook2)
          break
  
  def get_agg_gradient(self):
    self.feature_maps = None
    H = torch.zeros((2975, self.feature_map_shape[0], self.feature_map_shape[1], self.feature_map_shape[2]), device=self.device)  # Aggregate gradient
    for epoch in range(30):
      self.logger.info(f"\nStarting gradient epoch {epoch+1}/30...")
      for i_iter, batch in enumerate(self.train_dataloader, 0):
          image, true_label,_, _, _, idx = batch
          image, true_label = image.to(self.device), true_label.to(self.device)
  
          # 1. Apply the patch to the image
          patched_image, patched_label = self.apply_patch(image,true_label,self.adv_patch)
  
          # 2. Forward pass
          output = self.model1.predict(patched_image,patched_label.shape)
  
          # 3. Compute CE loss
          loss = self.criterion.compute_gradloss(output, patched_label)
  
          # 4. Compute gradient w.r.t. patch and update patch
          self.model1.model.zero_grad()
          if self.adv_patch.grad is not None:
              self.adv_patch.grad.zero_()
          loss.backward(retain_graph=True)
          with torch.no_grad():
              self.adv_patch += self.epsilon * self.adv_patch.grad.data.sign()
              self.adv_patch.clamp_(0, 1)  # Keep pixel values in valid range
          grad_feature_map = self.feature_maps_adv.grad  # Only works if feature_maps.requires_grad=True
  
          # If not requires_grad, use autograd.grad instead
          if grad_feature_map is None:
              print("grad_feature_map is None")
              grad_feature_map = torch.autograd.grad(loss, self.feature_maps_adv, retain_graph=True)[0]
  
          # 5. Normalize and aggregate
          # grad_feature_map /= torch.norm(grad_feature_map, p=2, dim=(1,2,3), keepdim=True) + 1e-8
          for i in range(image.shape[0]):
            H[idx[i]] = grad_feature_map[i] if H[idx[i]] is None else H[idx[i]] + grad_feature_map[i].detach()
          self.logger.info(f" Sample number: {i_iter+1}/2975")
          # Optional: delete big tensors
          del patched_image, output, grad_feature_map
          torch.cuda.empty_cache()
  
      
    return H

  def train(self, H):
    epochs, iters_per_epoch, max_iters = self.epochs, self.iters_per_epoch, self.max_iters
    self.feature_maps_adv = None
    self.feature_maps_rand = None
    H /= torch.norm(H, p=2, dim=(2,3), keepdim=True) + 1e-8 
    self.logger.info(f'H.shape: {H.shape}')
    start_time = time.time()
    self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, iters_per_epoch))
    IoU = []
    for ep in range(self.start_epoch, self.end_epoch):
      self.current_epoch = ep
      self.metric.reset()
      total_loss = 0
      samplecnt = 0
      momentum = torch.tensor(0, dtype=torch.float32).to(self.device)
      for i_iter, batch in enumerate(self.train_dataloader, 0):
          self.current_iteration += 1
          samplecnt += batch[0].shape[0]
          image, true_label,_, _, _, idx = batch
          image, true_label = image.to(self.device), true_label.to(self.device)
          

              
          
          # Randomly place patch in image and label(put ignore index)
          patched_image_adv, patched_label_adv = self.apply_patch(image,true_label,self.adv_patch)
          patched_image_rand, patched_label_rand = self.apply_patch(image,true_label,self.rand_patch)
          # fig = plt.figure()
          # ax = fig.add_subplot(1,2,1)
          # ax.imshow(patched_image[0].permute(1,2,0).cpu().detach().numpy())
          # ax = fig.add_subplot(1,2,2)
          # ax.imshow(patched_label[0].cpu().detach().numpy())
          # plt.show()

          # Forward pass through the model (and interpolation if needed)
          output1 = self.model1.predict(patched_image_adv,patched_label_adv.shape)
          output2 = self.model2.predict(patched_image_rand,patched_label_rand.shape)
          #F = torch.zeros(( self.feature_map_shape[1], self.feature_map_shape[2]), device=self.device)
          for i in range(image.shape[0]):
            F = ((self.feature_maps_rand[i]-self.feature_maps_adv[i])*H[idx[i]]) + (H[idx[i]])**2
          #plt.imshow(output.argmax(dim =1)[0].cpu().detach().numpy())
          #plt.show()
          #break

          # Compute adaptive loss
          loss = self.criterion.compute_trainloss(F)
          #loss = self.criterion.compute_loss_direct(output, patched_label)
          total_loss += loss.item()
          #break

          ## metrics
          self.metric.update(output1, patched_label_adv)
          pixAcc, mIoU = self.metric.get()

          # Backpropagation
          self.model1.model.zero_grad()
          self.model2.model.zero_grad()
          if self.adv_patch.grad is not None:
            self.adv_patch.grad.zero_()
          loss.backward()
          with torch.no_grad():
              #self.patch += self.epsilon * self.patch.grad.sign()  # Update patch using FGSM-style ascent
              grad = self.adv_patch.grad
              norm_grad = grad/ (torch.norm(grad) + 1e-8)
              momentum = (0.9*momentum) + norm_grad
              self.adv_patch += self.epsilon * momentum.sign()
              #self.patch += self.epsilon * self.patch.grad.data.sign()
              self.adv_patch.clamp_(0, 1)  # Keep pixel values in valid range

          ## ETA
          eta_seconds = ((time.time() - start_time) / self.current_iteration) * (iters_per_epoch*epochs - self.current_iteration)
          eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

          if i_iter % self.log_per_iters == 0:
            self.logger.info(
              "Epochs: {:d}/{:d} || Samples: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                  self.current_epoch, self.end_epoch,
                  samplecnt, self.batch_train*iters_per_epoch,
                  #self.optimizer.param_groups[0]['lr'],
                  self.epsilon,
                  loss.item(),
                  mIoU,
                  str(datetime.timedelta(seconds=int(time.time() - start_time))),
                  eta_string))
          

      average_pixAcc, average_mIoU = self.metric.get()
      average_loss = total_loss/len(self.train_dataloader)
      self.logger.info('-------------------------------------------------------------------------------------------------')
      self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
        self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))
      pickle.dump( self.adv_patch, open(self.config.experiment.log_patch_address+self.config.model.name+"_bbfa_modifiedloss"+".p", "wb" ) )
      
      #self.test() ## Doing 1 iteration of testing
      self.logger.info('-------------------------------------------------------------------------------------------------')
      #self.model.train() ## Setting the model back to train mode
      # if self.lr_scheduler:
      #     self.scheduler.step()

      IoU.append(self.metric.get(full=True))

    return self.patch.detach(),np.array(IoU)  # Return adversarial patch and IoUs over epochs

    
