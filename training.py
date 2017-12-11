# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Trainer for MRI amyloid data

"""
# Standard-library imports
import time

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm


class Trainer(object):
    """
    Callable wrapper which trains the model

    Args:
        model (emeramyloid.models.*): Wrapper over a pytorch model with
            additional parameters (e.g., best_acc)
        dataloaders (emeramyloid.data.dataloader.Dataloader): callable with
            pytorch Dataloaders
        model_performance (object): object with performance info (acc score etc)

    """
    def __init__(self, model, dataloader, criterion=None, optimizer=None, 
                 scheduler=None, num_epochs=50, use_cuda=True):
        self.model = model
        self.dataloader = dataloader

        self.criterion = criterion or nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        self.optimizer = optimizer or optim.SGD(self.model.M.parameters(), 
                                                lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = scheduler or lr_scheduler.StepLR(self.optimizer, 
                                                          step_size=7, 
                                                          gamma=0.1)

        self.num_epochs = num_epochs
        self.use_cuda = use_cuda

    def train(self):
        """Train model"""
        since = time.time()

        best_model_wts = self.model.M.state_dict()

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.M.train(True)  # Set model to training mode
                else:
                    self.model.M.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                num_images = len(self.dataloader.dataloders[phase])
                for data in tqdm(self.dataloader.dataloders[phase], 
                                 desc="Image dataloader", total=num_images):
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if self.use_cuda:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    outputs = self.model.M(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataloader.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataloader.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > self.model.best_acc:
                    self.model.best_acc = epoch_acc
                    best_model_wts = self.model.M.state_dict()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.model.best_acc))

        # load best model weights
        self.model.M.load_state_dict(best_model_wts)

        # save best accuracy and model configuration
        self.model.save_model()

        return self.model
