import pandas as pd
from exp_base import Exp_Base

from data.math_modeling_loader import Dataset_Train, Dataset_Pred
from torch.utils.data import DataLoader

import torch.nn as nn

from model import InformerStack

import os
import time
import numpy as np

from tools import EarlyStopping, adjust_learning_rate, StandardScaler

from torch import optim
import torch
from metrics import metric

class Exp_Math_Modeling(Exp_Base):
    def __int__(self,args):
        super(Exp_Math_Modeling,self).__init__(args) #子类Exp_Informer调用父类的构造函数

    # <-----对父类方法的重写
    def build_model(self):
        # model.model中定义的两个模型
        model_dict = {
            'informerstack': InformerStack,
        }
        # 生成模型model().float()
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers  # model == 'informer'就前面=;args.e_layers=num of encoder layers; args.s_layers=num of stack encoder layers
            # 传入参数，生成模型，.float()：将浮点参数和缓冲的类型转换为float类型（float32）
            model = model_dict[self.args.model](
                self.args.enc_in,self.args.dec_in,self.args.c_out,self.args.seq_len,self.args.label_len,
                self.args.pred_len,self.args.factor,self.args.d_model,self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,self.args.d_ff,self.args.dropout,self.args.attn,self.args.embed,
                self.args.freq,self.args.activation,self.args.output_attention,self.args.distil,
                self.args.mix,self.device
            ).float()
        # 是否使用多个gpu训练
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)  # 使用多个GPU

        return model

    def train(self,setting):
        train_data, train_loader = self.get_data(flag='train')
        vali_data, vali_loader = self.get_data(flag='val')
        test_data, test_loader = self.get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        # 早停法保存最佳模型参数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # 选择优化函数和误差标准
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp: #使用自动混合精度训练 https://zhuanlan.zhihu.com/p/165152789
            scaler = torch.cuda.amp.GradScaler()
            #scaler的大小在每次迭代中动态的估计，在每次scaler.step(optimizer)中，都会检查是否又inf或NaN的梯度出现：

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train() #Turn on the train mode
            epoch_time = time.time()
            print(">>>>>>>start  epoch:{0} training".format(epoch + 1))
            #1 先mini_batch
            #2 梯度清零
            #3 数据送进模型，获得预测结果
            #4 预测结果和真实值做损失函数
            #5 反向传播，更新参数
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad() #做随机梯度下降；将模型的参数梯度初始化为0
                #Forward pass: Compute predicted y by passing x to the model
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # print(pred, true)
                #Compute and print loss
                loss = criterion(pred, true)  #MSE 预测值和真实值做损失函数
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    print("<<<<<<<<<<<<<<<<<end")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward() # loss反向传播
                    model_optim.step() # 反向传播后参数更新

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))


            train_loss = np.average(train_loss) #求得所有iters的平均loss
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            print("\n")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))#用来加载torch.save() 保存的模型文件;load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中

        return self.model




    def get_data(self, flag):
        args = self.args
        # data_dict = {
        #     "taxi-D-201911-201912-15T" : Dataset_Yellow_Taxi,
        #     "监测点A-CO监测浓度-T" : Dataset_Yellow_Taxi,
        # }
        Data = Dataset_Train#data_dict[self.args.data]

        timeenc = 0 if args.embed != 'timeF' else 1

        if flag=='test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:  #train shuffle_flag = False ->mae:0.3910451829433441, mse:0.27577683329582214  True ->mae:0.3428313136100769, mse:0.23028385639190674
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        #初始化数据对象
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(" flag= ", flag, " len(data_set)= ", len(data_set))

        # torch.utils.data.DataLoader(dataset,batch_size,shuffle,drop_last,num_workers)
        # dataset： 加载torch.utils.data.Dataset对象数据,batch_size： 每个batch的大小
        # shuffle：是否对数据进行打乱,drop_last：是否对无法整除的最后一个datasize进行丢弃
        # um_workers：表示加载的时候子进程数，一般GPU使用
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()  # 在利用原始.pth模型进行前向推理之前，一定要先进行model.eval()操作，不启用 BatchNormalization 和 Dropout。https://blog.csdn.net/wuqingshan2010/article/details/106013660
        total_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self.get_data(flag='test')
        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mae:{}, mse:{}'.format(mae, mse))
        print("\n")

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    # 对父类方法的重写----->
    def predict(self, setting, load=False):
        pred_data, pred_loader = self.get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _select_optimizer(self):
        #优化函数，model.parameters()为该实例中可优化的参数，lr为参数优化的选项
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion =  nn.MSELoss() #平均平方误差，简称均方误差。
        return criterion

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device) #torch.Size([32, 96, 51])
        batch_y = batch_y.float()  #torch.Size([32, 72, 51])

        batch_x_mark = batch_x_mark.float().to(self.device) #torch.Size([32, 96, 5])
        batch_y_mark = batch_y_mark.float().to(self.device) #torch.Size([32, 72, 5])

        # decoder input
        #torch.zeros()返回一个由标量值0填充的张量，其形状由变量参数size定义
        #torch.ones()返回一个由标量值1填充的张量，其形状由变量参数size定义
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        #dec_inp:torch.Size([32, 24, 51])
        #torch.cat([x1,x2], dim=1);选择第二维扩维
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp: #False; use automatic mixed precision training
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention: #False; whether to output attention in ecoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse: #False; inverse output data
            outputs = dataset_object.inverse_transform(outputs)

        # outputs_inverse_transform = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features=='MS' else 0 #forecasting task, options:[M, S, MS];
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y #(32,24,51); (32,24,51)

    def showResult(self, setting):
        test_data, test_loader = self.get_data(flag='test')
        # print('预测长度:')
        path = os.path.join('results/', setting)

        res_metrics = np.load(os.path.join(path, 'metrics.npy'))
        res_metrics = res_metrics.tolist()
        print("res_metrics : (mae, mse, rmse, mape, mspe) {}".format(res_metrics))

        res_pred = np.load(os.path.join(path, 'pred.npy'))
        res_pred_len = res_pred.shape[1]
        res_pred = res_pred.reshape(-1, res_pred.shape[-2], res_pred.shape[-1])
        res_pred = test_data.inverse_transform(res_pred).tolist()

        res_pred = res_pred[-res_pred_len]

        res_true = np.load(os.path.join(path, 'true.npy'))
        res_true = res_true.reshape(-1, res_true.shape[-2], res_true.shape[-1])
        res_true = test_data.inverse_transform(res_true).tolist()

        res_true = res_true[-res_pred_len]

        pd_raw_data = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))

        res_true_df = pd.DataFrame(res_true, index=pd_raw_data['date'][-res_pred_len:])
        res_pred_df = pd.DataFrame(res_pred, index=pd_raw_data['date'][-res_pred_len:])

        cols = list(pd_raw_data.columns)
        cols.remove(self.args.target)
        cols.remove('date')

        # print(res_pred_df)

        res_pred_df.columns = cols + [self.args.target]
        res_true_df.columns = cols + [self.args.target]

        if self.args.features == 'S' or self.args.features == 'MS':
            res_pred_df = res_pred_df[self.args.target]
            res_true_df = res_true_df[self.args.target]

        pred_true_pd = pd.concat([res_pred_df, res_true_df], axis=1)

        save_floder = 'res_pred_true/'
        save_path = os.path.join(save_floder, setting)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pred_true_pd.to_csv(os.path.join(save_path, 'pred_true.csv'))

        print('true_pred result->{}'.format(save_path))

        if os.path.exists(os.path.join(path, 'real_prediction.npy')):
            real_pred = np.load(os.path.join(path, 'real_prediction.npy'))
            # print(real_pred)
            # real_pred = real_pred.reshape(real_pred.shape[0], real_pred.shape[-1])
            pred_data, pred_loader = self.get_data(flag='pred')
            tmp_stamp = pd_raw_data[['date']][-1:]
            tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
            pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.args.pred_len + 1, freq=self.args.freq)
            # real_pred = pred_data.inverse_transform(real_pred).tolist()
            real_pred = np.array(pred_data.inverse_transform(real_pred))
            print(real_pred.shape)
            real_pred = real_pred.reshape(real_pred.shape[-2], real_pred.shape[-1])


            real_pred_df = pd.DataFrame(real_pred, index=pred_dates[1:], columns=cols + [self.args.target])

            if self.args.features == 'M':
                print("预测结果：\n", real_pred_df)
                real_pred_df.to_csv(os.path.join(save_path, 'real_pred.csv'))
            else:
                print("预测结果：\n", real_pred_df[self.args.target])
                real_pred_df[self.args.target].to_csv(os.path.join(save_path, 'real_pred.csv'))

        return res_metrics


