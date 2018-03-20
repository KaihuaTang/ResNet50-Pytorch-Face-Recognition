import os
import torch
import numpy as np
from data import *
from ResNet import *
from VGG import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms

class Train:
    def __init__(self, root_path = "CACD2000/", model_name = "resnet50", number_classes = 2000, path="model.pkl", loadPretrain=0):
        """
        Init Dataset, Model and others
        """
        self.save_path = path
        self.cacd_dataset = ImageData(root_path=root_path, label_path="data/label.npy", name_path="data/name.npy", train_mode = "train")
        if model_name == "resnet50":
            self.model = resnet50(pretrained=(loadPretrain==1), num_classes = number_classes, model_path = path)
        elif model_name == "vgg16":
            self.model = vgg16(pretrained=(loadPretrain==1), num_classes = number_classes, model_path = path)

        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
           self.model.cuda()

    def start_train(self, epoch=10, batch_size=32, learning_rate=0.001, batch_display=50, save_freq=1):
        """
        Detail of training
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epoch_num):
            epoch_count = 0
            total_loss = 0
            dataloader = DataLoader(self.cacd_dataset, batch_size=self.batch_size, shuffle=True) # num_workers=8
            
            for i_batch, sample_batch in enumerate(dataloader):
 
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                """
                for i in range(images_batch.shape[0]):
                    img_tmp = transforms.ToPILImage()(images_batch[i]).convert('RGB')
                    plt.imshow(img_tmp)
                    plt.pause(0.001)
                """                   
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                    target_label = autograd.Variable(labels_batch.cuda(async=True))
                else:
                    input_image = autograd.Variable(images_batch)
                    target_label = autograd.Variable(labels_batch)
                # Step.2 calculate loss
                output = self.model(input_image)
                loss = loss_function(output, target_label)
                epoch_count += 1
                total_loss += loss
                # Step.3 Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Check Result
                if i_batch % batch_display == 0:
                    pred_prob, pred_label = torch.max(output, dim=1)
                    print("Input Label : ", target_label[:4])
                    print("Output Label : ", pred_label[:4])
                    batch_correct = (pred_label == target_label).sum().data[0] * 1.0 / self.batch_size
                    print("Epoch : %d, Batch : %d, Loss : %f, Batch Accuracy %f" %(epoch, i_batch, loss, batch_correct))
            """
            Save model
            """
            print("Epoch %d Average Loss : %f" %(epoch, total_loss * self.batch_size / epoch_count))
            if epoch % save_freq == 0:
                torch.save(self.model.state_dict(), self.save_path) 
    
