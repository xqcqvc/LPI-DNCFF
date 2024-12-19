import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, \
    MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, MaxPool2D, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
import argparse
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import Reshape, LSTM
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import *
from keras.layers import Reshape
from keras.callbacks import EarlyStopping
from keras.layers import Dot, Activation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 对数据进行标准化或归一化
def preprocess_data(X, scaler=None, stand=True):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X


# 评价指标
def calculate_performance(pred_res, pred_label, test_label):
    tn, fp, fn, tp = metrics.confusion_matrix(test_label, pred_label).ravel()
    auc = metrics.roc_auc_score(y_score=pred_res, y_true=test_label)
    ap = metrics.average_precision_score(y_score=pred_res, y_true=test_label)
    acc = metrics.accuracy_score(y_pred=pred_label, y_true=test_label)
    mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=test_label)
    f1_score = metrics.f1_score(y_pred=pred_label, y_true=test_label)
    sensitive = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)

    print()
    print('test result')
    print('acc', acc)
    print('auc', auc)
    print('mcc', mcc)
    print('f1-score', f1_score)
    print('sensitive', sensitive)
    print('specificity', specificity)
    print('ppv', ppv)
    print('ap', ap)
    return acc, auc, mcc, f1_score, sensitive, specificity, ppv, ap

# 标签处理
def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label

# 对标签数据进行预处理
def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = to_categorical(y)
    return y, encoder


# 将lncRNA序列转换为DNVFF编码
def get_RNA_seq_concolutional_array(seq, motif_len=4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([1/4] * 4)
    for i in range(row - 3, row):
        new_array[i] = np.array([1/4] * 4)
    for i in range(len(seq) - 1):
        val = seq[i]
        two_chars = seq[i:i + 2]
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0] * 4)
            continue
        try:
            bicoding_dict = {'AC': [1, 0, 1, 4], 'GC': [-1, 0, 2, 4], 'TC': [0, 1, 3, 4], 'CC': [0, -1, 4, 4],
                             'AT': [1, 0, 1, 3], 'GT': [-1, 0, 2, 3], 'TT': [0, 1, 3, 3], 'CT': [0, -1, 4, 3],
                             'AG': [1, 0, 1, 2], 'GG': [-1, 0, 2, 2], 'TG': [0, 1, 3, 2], 'CG': [0, -1, 4, 2],
                             'AA': [1, 0, 1, 1], 'GA': [-1, 0, 2, 1], 'TA': [0, 1, 3, 1], 'CA': [0, -1, 4, 1]}
            if two_chars in bicoding_dict:
                new_array[i] = np.array(bicoding_dict[two_chars])
        except:
            pdb.set_trace()
    return new_array

# 将lncRNA序列转换为独热编码
def get_RNA_seq_concolutional_array2(seq, motif_len=4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)
    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    return new_array

# 将蛋白质序列转换为CFF编码
def get_pro_seq_concolutional_array(seq, motif_len=5):
    alpha = 'PQRYWTMNVLHSFCIKADG'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 5))
    for i in range(motif_len - 1):
        new_array[i] = np.array([1/5] * 5)
    for i in range(row - 3, row):
        new_array[i] = np.array([1/5] * 5)
    for i in range(len(seq) - 1):
        val = seq[i]
        i = i + motif_len - 1
        if val not in 'PQRYWTMNVLHSFCIKADG':
            new_array[i] = np.array([0] * 5)
            continue
        bicoding_dict = {'P': [0, -1, 4, 4, 1], 'W': [0, 1, 3, 2, 0], 'V': [-1, 0, 2, 3, 0], 'F': [0, 1, 3, 3, 1],
                         'A': [-1, 0, 3, 1, 1],
                         'Q': [0, -1, 4, 1, 0], 'T': [1, 0, 1, 4, 0], 'L': [0, -1, 4, 3, 1], 'C': [0, 1, 3, 2, 1],
                         'D': [-1, 0, 2, 1, 0],
                         'R': [0, -1, 4, 2, 0], 'M': [1, 0, 1, 3, 1], 'H': [0, -1, 4, 1, 1], 'I': [1, 0, 1, 3, 0],
                         'G': [-1, 0, 2, 2, 0],
                         'Y': [0, 1, 3, 1, 0], 'N': [1, 0, 1, 1, 1], 'S': [0, 1, 3, 4, 1], 'K': [1, 0, 1, 1, 0],
                         'E': [-1, 0, 2, 1, 1]}
        if val in bicoding_dict:
            new_array[i] = np.array(bicoding_dict[val])
    return new_array

# 将蛋白质序列转换为独热编码
def get_pro_seq_concolutional_array2(seq, motif_len=7):
    alpha = 'AIYHRDC'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 7))
    for i in range(motif_len - 1):
        new_array[i] = np.array([1/7] * 7)
    for i in range(row - 3, row):
        new_array[i] = np.array([1/7] * 7)
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'AIYHRDC':
            new_array[i] = np.array([1/7] * 7)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    return new_array

# 对序列进行填充，使其达到指定的最大长度
def padding_sequence_new(seq, max_len=101, repkey='N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    return new_seq


# 将给定序列按照指定的窗口大小进行分割,并确保窗口之间有一定的重叠部分
def split_overlap_seq(seq, window_size=101):
    overlap_size = 20
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size) // (window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size) % (window_size - overlap_size)
    else:
        num_ins = 0
    end = 0
    for ind in range(num_ins):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1, max_len=window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, max_len=window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs


# 将LncRNA序列数据转换为特征矩阵
def get_lncbag_data(data, channel=7, window_size=101):
    bags = []
    seqs = data["seq"]
    for index, seq in enumerate(seqs):
        bag_seqs = split_overlap_seq(seq, window_size=window_size)
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)

        if num_of_ins > channel:
            start = (num_of_ins - channel) // 2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) < channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                tri_fea = get_RNA_seq_concolutional_array('N' * window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags


# 将蛋白质序列数据转换特征矩阵
def get_probag_data(data, channel=7, window_size=101):
    bags = []
    seqs = data["seq"]
    for index, seq in enumerate(seqs):
        bag_seqs = split_overlap_seq(seq, window_size=window_size)
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_pro_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        if num_of_ins > channel:
            start = (num_of_ins - channel) // 2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) < channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                tri_fea = get_pro_seq_concolutional_array('N' * window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags


# 从给定的FASTA格式文件中读取LncRNA序列数据，并将其存储在列表中
def read_lncseq_graphprot(seq_file):
    seq_list = []
    name = []
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name.append(line[1:-1])
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
    return seq_list, name


# 从给定的FASTA格式文件中读取蛋白质序列数据，并将其存储在列表中
def read_proseq_graphprot(seq_file):
    seq_list = []
    name = []
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name.append(line[1:-1])
            else:
                seq = line[:-1]
                seq_list.append(seq)
    return seq_list, name


# 读取lncRNA的数据文件，并将其处理成一个字典形式的数据结构
def read_lncdata_file(dataall, train=True):
    data = dict()
    seqs, name = read_lncseq_graphprot(dataall)
    data["seq"] = seqs
    return data, name


# 读取蛋白质序列的数据文件，并将其处理成一个字典形式的数据结构
def read_prodata_file(dataall, train=True):
    data = dict()
    seqs, name = read_proseq_graphprot(dataall)
    data["seq"] = seqs
    return data, name


def get_lncdata(data, channel=7, window_size=101, train=True):
    data, name = read_lncdata_file(data, train=train)
    if channel == 1:
        train_bags = get_bag_lncdata_1_channel(data, max_len=window_size)
    else:
        train_bags = get_lncbag_data(data, channel=channel, window_size=window_size)
    return train_bags, name


def get_prodata(data, channel=7, window_size=101, train=True):
    data, name = read_prodata_file(data, train=train)
    if channel == 1:
        train_bags = get_bag_prodata_1_channel(data, max_len=window_size)
    else:
        train_bags = get_probag_data(data, channel=channel, window_size=window_size)
    return train_bags, name


# 用于对序列进行填充，以确保序列达到指定的最大长度,用于单通道的情况
def padding_sequence(seq, max_len=501, repkey='N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq


def get_bag_lncdata_1_channel(data, max_len=501):
    bags = []
    seqs = data["seq"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len=max_len)
        bag_subt = []
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags



def get_bag_prodata_1_channel(data, max_len=501):
    bags = []
    seqs = data["seq"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len=max_len)
        bag_subt = []
        tri_fea = get_pro_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags


def get_strclncdata(lncfea, dataset):
    lncDic = {}
    for index, i in enumerate(lncfea):
        if i.startswith('\n'):
            continue
        L = i[:-1].split('\t')
        lncDic[L[0]] = L[1:]
    return lncDic


def get_strcprodata(profea, dataset):
    proDic = {}
    for index, i in enumerate(profea):
        L = i[:-1].split('\t')
        proDic[L[0]] = L[1:]
    return proDic

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


def LPIDNCFF(lnc_window_size, pro_window_size, lnc_glowindow, pro_glowindow,
             train_lnc_data, train_pro_data, train_glolnc_data, train_glopro_data, train_strc1_data, train_strc2_data,
             Y_train,
             val_lnc_data, val_pro_data, val_glolnc_data, val_glopro_data, val_strc1_data, val_strc2_data,
             Y_val,
             test_lnc_data, test_pro_data, test_glolnc_data, test_glopro_data, test_strc1_data, test_strc2_data,
             Y_test,
             real_labels):
    # locCNN,局部特征提取部分的输入，分别对应着lncRNA和protein的局部特征
    inx1 = Input(shape=(7, 4, lnc_window_size + 6))
    inx2 = Input(shape=(7, 5, pro_window_size + 8))
    filter1 = 16
    filter2 = 32
    kernel_size1 = (4, 30)
    kernel_size2 = (5, 30)
    dense1 = 32
    #GloCNN,全局特征提取部分的输入，分别对应着lncRNA和protein的全局特征
    inx3 = Input(shape=(1, 4, lnc_glowindow + 6))
    inx4 = Input(shape=(1, 5, pro_glowindow + 8))
    filter3 = 32
    filter4 = 64
    kernel_size3 = (4, 40)
    kernel_size4 = (5, 40)
    dense3 = 64

    inx5 = Input(shape=(117,))
    inx6 = Input(shape=(2764,))

    # local
    # Convolution layer
    #lnc
    x1 = Conv2D(filters=filter1, kernel_size=kernel_size1, strides=1, padding='same', data_format='channels_first')(
        inx1)
    x1 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPool2D(pool_size=4, strides=4, data_format='channels_first')(x1)
    x1 = Conv2D(filters=filter2, kernel_size=kernel_size1, strides=1, padding='same', data_format='channels_first')(x1)
    x1 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x1)
    x1 = Activation('relu')(x1)
    x1 = GlobalMaxPooling2D(data_format='channels_first')(x1)
    x1 = Dropout(0.2)(x1)
    
    #pro
    x2 = Conv2D(filters=filter1, kernel_size=kernel_size2, strides=1, padding='same', data_format='channels_first')(
        inx2)
    x2 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPool2D(pool_size=5, strides=5, data_format='channels_first')(x2)
    x2 = Conv2D(filters=filter2, kernel_size=kernel_size2, strides=1, padding='same', data_format='channels_first')(x2)
    x2 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x2)
    x2 = Activation('relu')(x2)
    x2 = GlobalMaxPooling2D(data_format='channels_first')(x2)
    x2 = Dropout(0.2)(x2)
    #xlocal = Concatenate(axis=1)([x1, x2])
    xlocal = Concatenate()([x1, x2])
    xlocal = Dense(dense1)(xlocal)

    # global
    # Convolution layer
    #lnc
    x3 = Conv2D(filters=filter3, kernel_size=kernel_size3, strides=1, padding='same', data_format='channels_first')(
        inx3)
    x3 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPool2D(pool_size=4, strides=4, data_format='channels_first')(x3)
    x3 = Conv2D(filters=filter4, kernel_size=kernel_size3, strides=1, padding='same', data_format='channels_first')(x3)
    x3 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x3)
    x3 = Activation('relu')(x3)
    x3 = GlobalMaxPooling2D(data_format='channels_first')(x3)
    x3 = Dropout(0.2)(x3)
    
    #pro
    x4 = Conv2D(filters=filter3, kernel_size=kernel_size4, strides=1, padding='same', data_format='channels_first')(
        inx4)
    x4 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x4)
    x4 = Activation('relu')(x4)
    x4 = MaxPool2D(pool_size=5, strides=5, data_format='channels_first')(x4)
    x4 = Conv2D(filters=filter4, kernel_size=kernel_size4, strides=1, padding='same', data_format='channels_first')(x4)
    x4 = BatchNormalization(epsilon=1e-06, momentum=0.9)(x4)
    x4 = Activation('relu')(x4)
    x4 = GlobalMaxPooling2D(data_format='channels_first')(x4)
    x4 = Dropout(0.2)(x4)
    # Concatenate
    xglobal = Concatenate()([x3, x4])
    xglobal = Dense(dense3)(xglobal)

    #FC
    xstrc1 = Dense(64, activation='relu')(inx5)
    xstrc1 = Dense(32, activation='relu')(xstrc1)
    xstrc2 = Dense(512, activation='relu')(inx6)
    xstrc2 = Dense(128, activation='relu')(xstrc2)
    xstrc = Concatenate()([xstrc1, xstrc2])
    xstrc = Dense(32)(xstrc)

    x = Concatenate()([xlocal, xglobal, xstrc])
    #x = Concatenate()([xlocal, xglobal])
    #x = xstrc

    x_reshaped = Reshape((1, -1))(x)

    # BiLSTM层
    #lstm_units = 64
    #bilstm_output = Bidirectional(LSTM(lstm_units, return_sequences=True))(x_reshaped)
    #lstm_output = LSTM(lstm_units, return_sequences=True)(x_reshaped)
    #flattened_output = Flatten()(lstm_output)
    
    #Transformer层
    num_heads = 4 # 多头注意力的头数
    ff_dim = 32   # 前馈神经网络的维度
    dropout_rate = 0.2
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x_reshaped, x_reshaped)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization()(attention_output)
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = LayerNormalization()(ff_output)
    flattened_output = Flatten()(ff_output)

    #Attention
    #attention_output = AttentionLayer()(flattened_output)
    
    #fully-connected layer
    #x = Dense(64)(x)
    #x = Activation('relu')(x)
    #x = Dropout(0.5)(x)

    xout = Dense(2, activation='softmax')(flattened_output)

    model = Model(inputs=[inx1, inx2, inx3, inx4,inx5, inx6], outputs=[xout])
    
    print(model.summary())

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #早停
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    print('Training --------------')

    model.fit(
        x=[train_lnc_data, train_pro_data, train_glolnc_data, train_glopro_data, train_strc1_data, train_strc2_data],
        y=Y_train, validation_data=(
        [val_lnc_data, val_pro_data, val_glolnc_data, val_glopro_data,val_strc1_data, val_strc2_data], Y_val),
        batch_size=256, epochs=100, shuffle=True, verbose=1, callbacks=[early_stopping])
    
    # test
    print('\nTesting---------------')

    loss, accuracy = model.evaluate(
        [test_lnc_data, test_pro_data, test_glolnc_data, test_glopro_data, test_strc1_data, test_strc2_data], Y_test,
        verbose=1)
    print(loss, accuracy)

    # get the confidence probability
    testres = model.predict(
        [test_lnc_data, test_pro_data, test_glolnc_data, test_glopro_data, test_strc1_data, test_strc2_data], verbose=1)
    pred_res = testres[:, 1]

    proba_res = transfer_label_from_prob(pred_res)
    test_label = [int(x) for x in real_labels]

    calculate_performance(pred_res, proba_res, test_label)
    
    feature_extractor = Model(inputs=model.input, outputs=flattened_output)
    train_features = feature_extractor.predict(
        [train_lnc_data, train_pro_data, train_glolnc_data, train_glopro_data, train_strc1_data, train_strc2_data], verbose=1)
    test_features = feature_extractor.predict(
        [test_lnc_data, test_pro_data, test_glolnc_data, test_glopro_data, test_strc1_data, test_strc2_data], verbose=1)

    # PCA和t-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    #pca = PCA(n_components=2)
    train_features_2d = pca.fit_transform(train_features)
    #train_features_2d = tsne.fit_transform(train_features)
    
    plt.figure(figsize=(12, 12))

    pos_samples = train_features_2d[Y_train.argmax(axis=1) == 1]
    neg_samples = train_features_2d[Y_train.argmax(axis=1) == 0]

    plt.scatter(pos_samples[:, 0], pos_samples[:, 1],  s=10,alpha=0.5, label='Positive Samples')
    plt.scatter(neg_samples[:, 0], neg_samples[:, 1],  s=10,alpha=0.5, label='Negative Samples')
    
    plt.scatter(train_features_2d[:, 0], train_features_2d[:, 1], c=Y_train.argmax(axis=1), cmap='viridis', alpha=1,s=10)
    plt.title('Train Features PCA')

    #plt.scatter(test_features_2d[:, 0], test_features_2d[:, 1], c=Y_test.argmax(axis=1), cmap='viridis', alpha=0.5)
    #plt.title('Test Features PCA')
    
    #plt.savefig('7317_Bilstm_PCA.png')

    return accuracy, model


def main(dataname):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.compat.v1.Session(config=config)

    if dataname == '7317' or dataname == 'ran7317':
        lnc = './Datasets/RPI7317/RNA_human_fasta.fasta'
        pro = './Datasets/RPI7317/protein_human_fasta.fasta'
        lncstrc = open('Datasets/RPI7317/lncRED.fasta', 'r').readlines()
        lncstrc4 = open('./Datasets/RPI7317/lncDNC.fasta', 'r').readlines()
        lncstrc5 = open('Datasets/RPI7317n/lnc3mer.fasta', 'r').readlines()
        prostrc = open('Datasets/RPI7317/proAAC.fasta', 'r').readlines()
        prostrc9 = open('Datasets/RPI7317n/pro3mer.fasta', 'r').readlines()
        prostrc10 = open('Datasets/RPI7317/pro4mer.fasta', 'r').readlines()
    elif dataname == '1847' or dataname == 'ran1847':
        lnc = './Datasets/RPI1847/RNA_mouse_fasta.fasta'
        pro = './Datasets/RPI1847/protein_mouse_fasta.fasta'
        lncstrc = open('Datasets/RPI1847/lncRED.fasta', 'r').readlines()
        lncstrc4 = open('./Datasets/RPI1847/lncDNC.fasta', 'r').readlines()
        lncstrc5 = open('Datasets/RPI1847/lnc3mer.fasta', 'r').readlines()
        prostrc = open('Datasets/RPI1847/proAAC.fasta', 'r').readlines()
        prostrc9 = open('Datasets/RPI1847/pro3mer.fasta', 'r').readlines()
        prostrc10 = open('Datasets/RPI1847/pro4mer.fasta', 'r').readlines()
    elif dataname == '21850' or dataname == 'ran21850':
        lnc = './Datasets/RPI21850/lncseq.fasta'
        pro = './Datasets/RPI21850/proseq.fasta'
        lncstrc = open('Datasets/RPI21850/lncRED.fasta', 'r').readlines()
        lncstrc4 = open('Datasets/RPI21850/lncDNC.fasta', 'r').readlines()
        lncstrc5 = open('./Datasets/RPI21850/lnckmer3.fasta', 'r').readlines()
        prostrc = open('Datasets/RPI21850/proAAC.fasta', 'r').readlines()
        prostrc9 = open('Datasets/RPI21850/pro3mer.fasta', 'r').readlines()
        prostrc10 = open('Datasets/RPI21850/pro4mer.fasta', 'r').readlines()
    elif dataname == 'ATH948':
        lnc = './Datasets/ATH948/Arabidopsis_rna.fasta'
        pro = './Datasets/ATH948/Arabidopsis_protein.fasta'
        strc = './Datasets/ATH948/ATH_data.npy'
    elif dataname == 'ZEA22133':
        lnc = './Datasets/ZEA22133/Zea_mays_rna.fasta'
        pro = './Datasets/ZEA22133/Zea_mays_protein.fasta'
        strc = './Datasets/ZEA22133/ZEA_data.npy'

    findlnclen = open(lnc, 'r').readlines()
    findprolen = open(pro, 'r').readlines()
    lnclen = 0
    lncnum = 0
    for i in findlnclen:
        if i.startswith('>'):
            continue
        lncnum += 1
        lnclen += len(i) - 1
    prolen = 0
    pronum = 0
    for i in findprolen:
        if i.startswith('>'):
            continue
        pronum += 1
        prolen += len(i) - 1

    print(lnclen)
    print(prolen)
    print(lnclen // lncnum)
    print(prolen // pronum)

    lnc_channel = 7
    pro_channel = 7
    lnc_window_size = lnclen // lncnum // 7
    pro_window_size = prolen // pronum // 7
    lnc_glochannel = 1
    pro_glochannel = 1
    lnc_glowindow = lnclen // lncnum
    pro_glowindow = prolen // pronum

    print(lnc_window_size, pro_window_size, lnc_glowindow, pro_glowindow)

    train_lnc, lnc_name = get_lncdata(lnc, channel=lnc_channel, window_size=lnc_window_size)
    train_pro, pro_name = get_prodata(pro, channel=pro_channel, window_size=pro_window_size)
    train_glolnc, lnc_gloname = get_lncdata(lnc, channel=lnc_glochannel, window_size=lnc_glowindow)
    train_glopro, pro_gloname = get_prodata(pro, channel=pro_glochannel, window_size=pro_glowindow)

    lncDic = {name: seq for name, seq in zip(lnc_name, train_lnc)}
    proDic = {name: seq for name, seq in zip(pro_name, train_pro)}
    glolncDic = {name: seq for name, seq in zip(lnc_name, train_glolnc)}
    gloproDic = {name: seq for name, seq in zip(pro_name, train_glopro)}

    if dataname == '21850':
        data = open('Datasets/Train_dataset/RPI21850.txt', 'r').readlines()
        seed = 29
    elif dataname == 'ran21850':
        data = open('Datasets/Train_dataset/ran21850.txt', 'r').readlines()
        seed = 29
    elif dataname == '7317':
        data = open('Datasets/Train_dataset/NPinter_human/RPI7317.txt', 'r').readlines()
        seed = 13
    elif dataname == 'ran7317':
        data = open('Datasets/Train_dataset/NPinter_human/ran7317.txt', 'r').readlines()
        seed = 13
    elif dataname == '1847':
        data = open('Datasets/Train_dataset/NPinter_mouse/RPI1847.txt', 'r').readlines()
        seed = 13
    elif dataname == 'ran1847':
        data = open('Datasets/Train_dataset/NPinter_mouse/ran1847.txt', 'r').readlines()
        seed = 13
    elif dataname == 'ran1847':
        data = open('Datasets/Train_dataset/NPinter_mouse/ran1847.txt', 'r').readlines()
        seed = 13
    elif dataname == 'ATH948':
        data = open('Datasets/lncRNA-protein/Arabidopsis_all1.txt', 'r').readlines()
        seed = 29
    elif dataname == 'ZEA22133':
        data = open('Datasets/lncRNA-protein/Zea_mays_all.txt', 'r').readlines()
        seed = 29
    elif dataname == 'RPI369':
        data = open('Datasets/lncRNA-protein/RPI369_all1.txt', 'r').readlines()
        seed = 29
    elif dataname == 'RPI2241':
        data = open('Datasets/lncRNA-protein/RPI2241_all1.txt', 'r').readlines()
        seed = 29
    elif dataname == 'RPI1807':
        data = open('Datasets/lncRNA-protein/RPI1807_all1.txt', 'r').readlines()
        seed = 13

    data_Lst = np.array([i.split() for i in data])
    dataset = data_Lst[:, 0:2]
    labels = data_Lst[:, 2]
    y, encoder = preprocess_labels(labels)

    X_train, X_test_a, Y_train, Y_test_a = train_test_split(dataset, y, test_size=0.3, stratify=y,
                                                            random_state=seed)

    X_test, X_val, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                    random_state=seed)

    del X_test_a, Y_test_a
    gc.collect()

    print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train), len(X_test), len(X_val)))

    dataall1 = get_strclncdata(lncstrc, dataset)
    dataall2 = get_strclncdata(lncstrc4, dataset)
    dataall3 = get_strclncdata(lncstrc5, dataset)
    dataall4 = get_strcprodata(prostrc10, dataset)
    dataall5 = get_strcprodata(prostrc, dataset)
    dataall6 = get_strcprodata(prostrc9, dataset)

    datalncall = [dataall1[i[0]] + dataall2[i[0]] + dataall3[i[0]] for i in dataset]
    datalncall = preprocess_data(datalncall)
    strclncdata = np.array(datalncall)
    dataproall = [dataall4[i[1]] + dataall5[i[1]] + dataall6[i[1]] for i in dataset]
    dataproall = preprocess_data(dataproall)
    strcprodata = np.array(dataproall)

    X_train1, X_test_a, Y_train, Y_test_a = train_test_split(strclncdata, y, test_size=0.3, stratify=y,
                                                             random_state=seed)
    X_test1, X_val1, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                      random_state=seed)
    del X_test_a, Y_test_a
    gc.collect()
    print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train1), len(X_test1), len(X_val1)))

    X_train2, X_test_a, Y_train, Y_test_a = train_test_split(strcprodata, y, test_size=0.3, stratify=y,
                                                             random_state=seed)
    X_test2, X_val2, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                      random_state=seed)
    del dataset, X_test_a, Y_test_a
    gc.collect()
    print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train2), len(X_test2), len(X_val2)))

    train_lnc_data = np.array([lncDic[i[0]] for i in X_train if i[0] in lncDic])
    train_pro_data = np.array([proDic[i[1]] for i in X_train if i[1] in proDic])
    val_lnc_data = np.array([lncDic[i[0]] for i in X_val if i[0] in lncDic])
    val_pro_data = np.array([proDic[i[1]] for i in X_val if i[1] in proDic])
    test_lnc_data = np.array([lncDic[i[0]] for i in X_test if i[0] in lncDic])
    test_pro_data = np.array([proDic[i[1]] for i in X_test if i[1] in proDic])

    train_glolnc_data = np.array([glolncDic[i[0]] for i in X_train if i[0] in glolncDic])
    train_glopro_data = np.array([gloproDic[i[1]] for i in X_train if i[1] in gloproDic])
    val_glolnc_data = np.array([glolncDic[i[0]] for i in X_val if i[0] in glolncDic])
    val_glopro_data = np.array([gloproDic[i[1]] for i in X_val if i[1] in gloproDic])
    test_glolnc_data = np.array([glolncDic[i[0]] for i in X_test if i[0] in glolncDic])
    test_glopro_data = np.array([gloproDic[i[1]] for i in X_test if i[1] in gloproDic])

    train_strc1_data = X_train1
    val_strc1_data = X_val1
    test_strc1_data = X_test1
    train_strc2_data = X_train2
    val_strc2_data = X_val2
    test_strc2_data = X_test2

    real_labels = []
    for val in Y_test:
        if val[0] == 1:
            real_labels.append(0)
        else:
            real_labels.append(1)

    val_label_new = []
    for val in Y_val:
        if val[0] == 1:
            val_label_new.append(0)
        else:
            val_label_new.append(1)

    train_label_new = []
    for val in Y_train:
        if val[0] == 1:
            train_label_new.append(0)
        else:
            train_label_new.append(1)

    accuracy1, model1 = LPIDNCFF(lnc_window_size, pro_window_size, lnc_glowindow, pro_glowindow,
                                 train_lnc_data, train_pro_data, train_glolnc_data, train_glopro_data,
                                 train_strc1_data, train_strc2_data, Y_train,
                                 val_lnc_data, val_pro_data, val_glolnc_data, val_glopro_data,
                                 val_strc1_data, val_strc2_data,Y_val,
                                 test_lnc_data, test_pro_data, test_glolnc_data, test_glopro_data,
                                 test_strc1_data, test_strc2_data,Y_test, real_labels)

    model1.save("./Models/" + dataname + ".h5")



parser = argparse.ArgumentParser(
    description="LPI-DNCFF：Predicting lncRNA-Protein Interactions Using a Hybrid Deep Learning Model with Dinucleotide-Codon Fusion Feature Encoding")
parser.add_argument('-dataset', type=str, help='RPI21850, RPI7317, RPI1847, ATH948, ZEA22133, ran1847, ran7317 or ran21850')
args = parser.parse_args()
datname = args.dataset
main(datname)
