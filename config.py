# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Configuration for the emeramyloid project

"""
# Standard library
import os


class DevConf(object):
    """
    Basic configuration class for development environment

    """
    def __init__(self):
        super(DevConf, self).__init__()

        # PATH TO TRAIN/VAL/TEST DATA
        self.data_path = \
            "/media/Warehouse/bigdata-muw/mri-data/stony-brook-cardiac-atlas/converted-images"
            # "/home/asan/code/muw-projects/med-images/emeri-transfer/data/image-folder"

        self.data_paths = {
            x : os.path.join(self.data_path, x)
            for x in ['train', 'test', 'val']
            }

        self.model_conf_path = \
            "/home/asan/code/muw-projects/med-images/cardiac-atlas/model_configs"
            

        self.model_conf_paths = {
            x : os.path.join(self.model_conf_path, x) + ".pth"
            for x in ['best']
            }
