# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Finetuned model for MRI based on Resnet

"""
# Third-party imports
import torch
import torch.nn as nn
import torchvision
from torchvision import models

# Cross-library imports
import config


conf = config.DevConf()



class FinetunedConvnet(object):
    """
    Fine tuned model based on resnet, the final layer is learnt. This is a
    callable that wraps torch model with additional parameters (e.g., best acc)

    Args:
        model_config (string, ["best", "new"]): model config to load: current
            best (reads from file), new (initializes on top of resnet)

    """
    def __init__(self, model_config="best", use_cuda=True):
        self.M = models.resnet18(pretrained=True)
        self.model_config = model_config
        self.use_cuda = use_cuda

        # replace final linear layer
        num_features = self.M.fc.in_features
        # self.M.fc = nn.Linear(num_features, 2)
        self.M.fc = nn.Linear(num_features, 4)

        # will be changed if model_config = "best"
        self.best_acc = 0.0

        if self.use_cuda:
            self.M.cuda()

        if self.model_config == "best":
            self.load_model()


    def load_model(self):
        """Loads model configuration"""
        try:
            config = torch.load(conf.model_conf_paths["best"])
            self.M.load_state_dict(config["model_state_dict"])
            self.best_acc = config["best_acc"]
        except FileNotFoundError:
            print("configuration not found. loading default model")
        except:
            raise

    def save_model(self):
        """Saves current configuration of the model as the current best"""
        try:
            config = { 
                "model_state_dict": self.M.state_dict(),
                "best_acc": self.best_acc
            }

            torch.save(config, conf.model_conf_paths["best"])

        except FileNotFoundError:
            print("configuration not found. loading default model")
        except:
            raise
