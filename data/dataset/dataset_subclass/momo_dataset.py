import os

import numpy as np
import pandas as pd
import datetime
from libcity.data.dataset import TrafficStateGridOdDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir


class MOMODataset(TrafficStateGridOdDataset):

    def __init__(self, config):
        super().__init__(config)
        self.feature_name = {'X': 'float', 'W': 'float', 'y': 'float', 'P': 'float'}
        self.load_poi = False # 不加载POI数据
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample) + '_' + str(self.load_poi)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'traffic_state_{}.npz'.format(self.parameters_str))

    def exter(self,ext_data): # 处理外部数据
        num_samples = ext_data.shape[0]
        # insert timeinday dayinweek
        data_list = []
        time_ind = (self.ext_timesolts - self.ext_timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D") # 计算时间
        time_in_day = np.tile(time_ind, [1, 1]).transpose((1, 0)) # 扩展时间维度
        data_list.append(time_in_day) # 将timeinday添加到数据列表中

        dayofweek = []
        for day in self.ext_timesolts.astype("datetime64[D]"): # 将时间戳截断到日期部分（忽略时分秒）
            dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday()) # 获取星期几
        day_in_week = np.zeros(shape=(num_samples, 7)) # 创建一个全0的矩阵，维度是(num_samples, 7)
        day_in_week[np.arange(num_samples), dayofweek] = 1 # 将对应的星期几位置置为1
        data_list.append(day_in_week) # 将dayinweek添加到数据列表中

        for i in range(ext_data.shape[1]): # 遍历外部数据的每一列
            data_ind = ext_data[:, i] # 提取第i列数据
            data_ind = np.tile(data_ind, [1, 1]).transpose((1, 0)) # tile:将数据沿行和列方向各复制一次,transpose:转置,确保确保每个特征是一个列向量（形状 (N, 1)），便于后续按列合并。

            data_list.append(data_ind) # 将特征数据添加到数据列表中，形如
            #[
            #     time_in_day,      # 形状 (N,1)
            #     day_in_week,      # 形状 (N,7)
            #     ext_feature_1,    # 形状 (N,1)
            #     ext_feature_2,    # 形状 (N,1)
            #     ...
            # ]
        data = np.concatenate(data_list, axis=-1) # 将所有特征数据按列拼接在一起，形成一个新的数组，形状为 (N, 1+7+ext_dim)
        return data
    
    def _generate_ext_data(self, ext_data):
        num_samples = ext_data.shape[0]
        ext_data = self.exter(ext_data) # 处理外部数据
        offsets = np.sort(np.concatenate((np.arange(-self.input_window - self.output_window + 1, 1, 1),)))
        min_t = abs(min(offsets))
        max_t = abs(num_samples - abs(max(offsets)))
        W = []
        for t in range(min_t, max_t):
            W_t = ext_data[t + offsets, ...]
            W.append(W_t)
        W = np.stack(W, axis=0)
        return W

    def _generate_data(self):
        """
        加载数据文件(.gridod)和外部数据(.ext)，以X, W, y的形式返回

        Returns:
            tuple: tuple contains:
                X(np.ndarray): 模型输入数据，(num_samples, input_length, ..., feature_dim) \n
                W(np.ndarray): 模型外部数据，(num_samples, input_length, ext_dim)
                y(np.ndarray): 模型输出数据，(num_samples, output_length, ..., feature_dim)
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()

        # 加载外部数据 0 distance 1~29 POI
        # POI distance
        data_list = []
        if os.path.exists(self.data_path + 'POI.csv'):
            distance = self.adj_mx
            #distance = np.tile(distance, [self.input_window, 1, 1]).reshape((self.input_window, self.num_nodes, self.num_nodes, 1)) #
            distance = distance.reshape((self.num_nodes, self.num_nodes, 1))
            data_list.append(distance)

            P = pd.read_csv(self.data_path + 'POI.csv')

            def minmax_norm(df):
                return (df - df.min()) / (df.max() - df.min())

            P = minmax_norm(P)
            P = P.values
            data_p = np.zeros((self.num_nodes, self.num_nodes, 2 * P.shape[1]))
            for i in range(P.shape[0]):
                p_i = P[i, :]
                for j in range(P.shape[0]):
                    p_j = P[j, :]
                    data_p[i][j] = np.hstack((p_i, p_j))
            #data_p = np.tile(data_p, [self.input_window, 1, 1, 1])#
            data_list.append(data_p)
            data_list = np.concatenate(data_list, axis=-1)
        ext_data = self._load_ext()  # （len_time, ext_dim)
        W = self._generate_ext_data(ext_data)

        # 加载基本特征数据 1 time in day 2~8 day in week 9~16 weather（说明天气有8种情况）
        X_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            print(np.sum(df>=1))
            print(np.sum(df == 0))
            # np.save('cd_taxi.npy',np.array(df))
            # chengdu
            data = self._add_external_information(df, ext_data)
            # nyc
            # data = self._add_external_information(df)
            X, y = self._generate_input_data(data)
            # x: (num_samples, input_length, input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            X_list.append(X)
            y_list.append(y)
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        # df = self._load_dyna(data_files[0]).squeeze()
        self._logger.info("Dataset created")
        self._logger.info("X shape: {}, W shape: {}, y shape: ".format(str(X.shape), str(W.shape), y.shape))
        # P = []
        # if self.load_poi:
        #     P = pd.read_csv(self.data_path + 'POI.csv')
        #     P = P.iloc[:, 2:]
        #     P = P.values
        #     P = np.tile(P, [self.batch_size, 1, 1, 1]).transpose((0, 3, 1, 2))
        #     # P: B,C,T,N
        return X, W, y, data_list

    def _split_train_val_test(self, X, W, y): # 划分训练集、验证集和测试集
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = X.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_eval = num_samples - num_test - num_train
        # train
        x_train, w_train, y_train = X[:num_train], W[:num_train], y[:num_train]

        # eval
        x_eval, w_eval, y_eval = X[num_train: num_train + num_eval], \
                                 W[num_train: num_train + num_eval], y[num_train: num_train + num_eval]
        # test
        x_test, w_test, y_test = X[-num_test:], W[-num_test:], y[-num_test:]

        # log
        self._logger.info(
            "train\tX: {}, W: {}, y: {}".format(str(x_train.shape), str(w_train.shape), str(y_train.shape)))
        self._logger.info("eval\tX: {}, W: {}, y: {}".format(str(x_eval.shape), str(w_eval.shape), str(y_eval.shape)))
        self._logger.info("test\tX: {}, W: {}, y: {}".format(str(x_test.shape), str(w_test.shape), str(y_test.shape)))
        return x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test

    def _load_cache_train_val_test(self): # 加载缓存数据
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test, P = \
            cat_data['x_train'], cat_data['w_train'], cat_data['y_train'], cat_data['x_eval'], cat_data['w_eval'], \
            cat_data['y_eval'], cat_data['x_test'], cat_data['w_test'], cat_data['y_test'], cat_data['p']

        self._logger.info(
            "train\tX: {}, W: {}, y: {}".format(str(x_train.shape), str(w_train.shape), str(y_train.shape)))
        self._logger.info("eval\tX: {}, W: {}, y: {}".format(str(x_eval.shape), str(w_eval.shape), str(y_eval.shape)))
        self._logger.info("test\tX: {}, W: {}, y: {}".format(str(x_test.shape), str(w_test.shape), str(y_test.shape)))

        return x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test, P

    def _generate_train_val_test(self): # 生成训练集、验证集和测试集
        X, W, y, P = self._generate_data()
        x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test = self._split_train_val_test(X, W, y)

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                w_train=w_train,
                y_train=y_train,
                x_test=x_test,
                w_test=w_test,
                y_test=y_test,
                x_eval=x_eval,
                w_eval=w_eval,
                y_eval=y_eval,
                p=P
            )
            self._logger.info('Saved at ' + self.cache_file_name)

        return x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test, P

    def get_data(self):
        # 加载数据集
        x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test, P = [], [], [], [], [], [], [], [], [], []
        if self.data is None:
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test, P = self._load_cache_train_val_test()
            else:
                x_train, w_train, y_train, x_eval, w_eval, y_eval, x_test, w_test, y_test, P = self._generate_train_val_test()

        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = w_train.shape[-1]
        self.scaler = self._get_scalar(self.scaler_type, x_train, y_train)
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        w_train[..., :self.output_dim] = self.scaler.transform(w_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_eval[..., :self.output_dim] = self.scaler.transform(x_eval[..., :self.output_dim])
        w_eval[..., :self.output_dim] = self.scaler.transform(w_eval[..., :self.output_dim])
        y_eval[..., :self.output_dim] = self.scaler.transform(y_eval[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        w_test[..., :self.output_dim] = self.scaler.transform(w_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])

        if len(P)==0:
            P = [i for i in range(len(x_train))]
        else:
            P = np.tile(P, [len(x_train), self.input_window, 1, 1, 1])
        self._logger.info('POI info ' + str(len(P)))

        train_data = list(zip(x_train, w_train, y_train, P))
        eval_data = list(zip(x_eval, w_eval, y_eval, P))
        test_data = list(zip(x_test, w_test, y_test, P))

        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是网格的个数，
        len_row是网格的行数，len_column是网格的列数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "ext_dim": self.ext_dim,
                "output_dim": self.output_dim, "len_row": self.len_row, "len_column": self.len_column,
                "num_batches": self.num_batches}
