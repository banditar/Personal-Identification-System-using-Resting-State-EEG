from glob import glob
from time import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import RepeatedStratifiedKFold

import cnn
import util
from ExperimentRecord import ExperimentRecord
PATH_SUBJECT_FOLDERS = r'/home/eric/Documents/UT/HACKATHON/data/SUBJECTS'
channels = range(4)
data_type = 'EC'
fs = 256
sample_time = 60
window_length_s = 0.5

parameters = {'data_path': '/media/hit/1/EEG_Personal_Identification/mnenpz/',  # resting state EEG数据路径
                'data_type': data_type,  # 'EO', 'EC', 'EO&EC'
                'BATCH_SIZE': 64, 'EPOCHS': 2000,
                'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'channels': channels,
                'ica_enable': True, 'conv_layers': 3, 'ica_output': 64, 'conv_filters': 32, 'fc1_output': 512, 'dropout': 0.5,
                'learn_rate': 3e-4, 'classes': 2,
                'window_length': window_length_s*fs, 'sliding_ratio': 0.25,  # 'window_length' === 'points'
                'setup_seed': 1000
                }

record = ExperimentRecord(
    extra='{}{}'.format(parameters['data_type'], len(parameters['channels'])))
record.append('parameters: {}'.format(parameters))
util.setup_seed(parameters['setup_seed'])

# 交叉验证
best_test_accuracy_list, threshold_list, frr_list, far_list, eer_list = [], [], [], [], []

test_start = 0
test_end = int(fs*60*0.3) #30% of data (each file) for test
train_data, train_labels, test_data, test_labels = [], [], [], []
# 读取npz
label = 0
subjects_path = PATH_SUBJECT_FOLDERS
subjects_folders = glob(subjects_path+ '/*')
#npzFiles = glob(parameters['data_path'] + '*.npz')
#for file in npzFiles:
for subject in subjects_folders:
    #npz = np.load(file)
    for sample in glob(subject+'/*txt'):
        data = np.genfromtxt(sample, delimiter = ',').transpose()     
        print('SIZE',data.shape)   
        data = data[:4, :min(fs*sample_time, data.shape[1])]
        if len(data > 0):
            resting_data = data
            train_data_seg = util.split_samples(resting_data[:, :test_start], parameters['window_length'],
                                                parameters['sliding_ratio'])
            train_data_seg.extend(util.split_samples(resting_data[:, test_end:], parameters['window_length'],
                                                parameters['sliding_ratio']))
            test_data_seg = util.split_samples(resting_data[:, test_start:test_end], parameters['window_length'],
                                                parameters['sliding_ratio'])


            train_labels_seg = [label] * len(train_data_seg)
            test_labels_seg = [label] * len(test_data_seg)

            train_data.extend(train_data_seg)
            train_labels.extend(train_labels_seg)
            test_data.extend(test_data_seg)
            test_labels.extend(test_labels_seg)

    label += 1
# 转换数据类型和维度，适应模型输入
train_data = np.array(train_data, dtype=np.float32)
train_data = np.expand_dims(train_data, 1)
train_data = train_data.transpose([0, 1, 3, 2])  # 将时间维度和信道维度交换，信道维度作为最后一维
train_labels = np.array(train_labels, dtype=np.longlong)
test_data = np.array(test_data, dtype=np.float32)
test_data = np.expand_dims(test_data, 1)
test_data = test_data.transpose([0, 1, 3, 2])  # 将时间维度和信道维度交换，信道维度作为最后一维
test_labels = np.array(test_labels, dtype=np.longlong)
# 构建模型和优化器
model = cnn.CNN(channels=len(parameters['channels']), points=parameters['window_length'],
                ica_enable=parameters['ica_enable'], conv_layers=parameters['conv_layers'],
                classes=parameters['classes'], ica_output=parameters['ica_output'],
                conv_filters=parameters['conv_filters'], fc1_output=parameters['fc1_output'],
                dropout=parameters['dropout']).to(parameters['DEVICE'])
optimizer = optim.Adam(model.parameters(), lr=parameters['learn_rate'])

train_loader = util.set_data_loader(train_data, train_labels, parameters['BATCH_SIZE'])
test_loader = util.set_data_loader(test_data, test_labels, parameters['BATCH_SIZE'])
best_test_accuracy, threshold, frr, far, eer = util.run(record, model, optimizer, parameters['DEVICE'],
                                                        train_data, train_labels, test_data, test_labels,
                                                        parameters['BATCH_SIZE'], parameters['EPOCHS'],
                                                        parameters['classes'])
best_test_accuracy_list.append(best_test_accuracy)
threshold_list.append(threshold)
frr_list.append(frr)
far_list.append(far)
eer_list.append(eer)

record.append('best_test_accuracy average: {}\tstd:{}'.format(np.mean(best_test_accuracy_list),
                                                                np.std(best_test_accuracy_list)))
record.append('threshold_list average: {}\tstd:{}'.format(np.mean(threshold_list), np.std(threshold_list)))
record.append('frr_list average: {}\tstd:{}'.format(np.mean(frr_list), np.std(frr_list)))
record.append('far_list average: {}\tstd:{}'.format(np.mean(far_list), np.std(far_list)))
record.append('eer_list average: {}\tstd:{}'.format(np.mean(eer_list), np.std(eer_list)))
