import os
import torch


class Exp_Base(object):
    def __init__(self,args):
        self.args = args
        self.device = self.acquire_device() #gpu/cpu设置
        self.model = self.build_model().to(self.device) #设置使用device

    def acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def build_model(self):
        raise NotImplementedError
        return None

    def get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
