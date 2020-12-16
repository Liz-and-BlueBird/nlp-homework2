import torch
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import collections
import os
import time
from past.builtins import raw_input


def load_vocab():
    """
    加载字典:data/cnews.vocab.txt
    :return: Vocab类型的字典变量
    """
    with open('data/cnews.vocab.txt', 'rb') as vocab_file:
        # 分割字符组成列表
        review = vocab_file.read().decode('utf-8').split('\r\n')
        # 封装成Counter类型
        counter = collections.Counter(review)
        # 转换成Vocab类型
        return Vocab.Vocab(counter)


def load_category():
    """
    加载类别:data/cnews.category.txt
    :return: 建立dictionary类型的索引
    """
    with open('data/cnews.category.txt', 'rb') as category_file:
        # 分割类别组成列表
        review = category_file.read().decode('utf-8').split(', ')
        # 构建dictionary类型的索引
        return {item: i - 1 for i, item in enumerate(review, start=1)}


def load_data():
    """
    加载数据:data/cnews.train(val,test).txt
    :return: list[label,sentence](训练集),list[...](验证集),list[...](测试集)
    """

    def read_data(file):
        """
        读取数据
        :param file: 从file文件中读取
        :return: list[label,sentence]
        """
        with open(file, 'rb') as read_file:
            # 去空格
            data = read_file.read().decode('utf-8').replace(' ', '').split('\n')
            # 封装成列表,尾部为空的一个元素丢弃
            return [[sentence[:2], sentence[3:]] for sentence in data][:-1]

    return read_data('data/cnews.train.txt'), read_data('data/cnews.val.txt'), read_data('data/cnews.test.txt')


def preprocess_data(data, vocab, category):
    """
    完成输入数据的预处理，经过索引化+padding，变成可以输入进网络的格式
    :param data: 标签+句子格式的列表,未索引化
    :param vocab: 字典
    :param category: 类别索引
    :return: 数据特征feature,数据标签label
    """
    # 规定句子长度为输入句子中最长者的长度
    length = len(max([x[1] for x in data]))

    def tokenized(x):
        """
        索引化
        :param x: 未索引化的标签+句子
        :return: 索引化后的标签+句子
        """
        return category[x[0]], [vocab.stoi[word] for word in x[1]]

    def padding(x):
        """
        裁剪或填充
        :param x: 索引化后长度不等的句子
        :return: 长度统一为length的句子
        """
        return x[:length] if len(x) > length else x + [0] * (length - len(x))

    # 特征,shape:(句子总数,句子长度length)
    feature = torch.tensor([padding(tokenized(item)[1]) for item in data])
    # 标签,shape:(句子总数)
    label = torch.tensor([tokenized(item)[0] for item in data])
    return feature, label


def load_batch_iter(feature, label, batch_size, shuffle=True):
    """
    返回批量数据
    :param feature: 特征
    :param label: 标签
    :param batch_size: 批量大小
    :param shuffle: 数据是否搅乱,默认为True
    :return: 批量数据迭代器
    """
    data_set = Data.TensorDataset(feature, label)
    data_iter = Data.DataLoader(data_set, batch_size, shuffle=shuffle)
    return data_iter


class GlobalMaxPool(nn.Module):
    """
    全局最大池化层
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x shape:(批量大小batch_size,输出通道数out_channels,句子长度length-卷积核宽度kernel_size+1)
        # return shape:(批量大小batch_size,输出通道数out_channels,1)
        return F.max_pool1d(x, kernel_size=x.shape[2])


class TextCNN(nn.Module):
    """
    TextCNN模型
    """

    def __init__(self, vocab_size, embed_size, num_channels, kernel_sizes):
        """
        :param vocab_size: 字典大小
        :param embed_size: 词向量维度 = 卷积层输入通道数
        :param num_channels: 卷积层输出通道数
        :param kernel_sizes: 卷积核宽度
        """
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 多个一维卷积层
        self.convs = nn.ModuleList()
        for x, y in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=embed_size, out_channels=x, kernel_size=y))
        # 最大池化层,卷积层共用
        self.pool = GlobalMaxPool()
        # 丢弃层
        self.dropout = nn.Dropout(0.5)
        # 全连接层
        self.decoder = nn.Linear(sum(num_channels), 10)

    def forward(self, inputs):
        # inputs shape:(批量大小batch_size,句子长度length)
        # 嵌入层输出:(批量大小batch_size,句子长度length,词向量维度embed_size)
        # 再经一次变换得到:(批量大小batch_size,词向量维度embed_size,句子长度length)
        embeddings = self.embedding(inputs).permute(0, 2, 1)
        # 每个卷积层经relu激活,最大池化后输出：(批量大小batch_size,输出通道数out_channels,1)
        # 再经过squeeze压缩大小为1的最后一维,得到:(批量大小batch_size,输出通道数out_channels)
        # 每个卷积层输出经cat在词向量维度上拼接在一起,得到:(批量大小batch_size,输出通道总数sum(out_channels))
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 丢弃层丢弃
        # 全连接层输出：(批量大小batch_size,类别个数)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def evaluate_accuracy(data_iter, net, device=None):
    """
    在数据集上验证模型
    :param data_iter: 数据集
    :param net: 神经网络
    :param device: cpu or gpu
    :return: 模型在数据集上的准确度
    """
    # 没有指定device则使用net的device
    if device is None:
        device = list(net.parameters())[0].device
    # 准确数,样本数
    acc_sum, n = 0., 0
    with torch.no_grad():
        # 进入评估模式
        net.eval()
        for X, y in data_iter:
            # 准确数累加
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            # 样本数累加
            n += y.shape[0]
        # 回到训练模式
        net.train()
    return acc_sum / n


def train(train_iter, val_iter, net, loss, optimizer, device, num_epochs):
    """
    训练网络
    :param train_iter: 训练集
    :param val_iter: 验证集
    :param net: 神经网络
    :param loss: 损失函数
    :param optimizer: 优化器
    :param device: cpu or gpu
    :param num_epochs: 迭代次数
    """
    # use cpu or gpu
    net = net.to(device)
    print("training on", device)
    for epoch in range(num_epochs):
        # 训练损失,训练准确数,批量数,训练样本数,一次迭代的开始时间
        train_loss_sum, train_acc_sum, batch_count, n, start = 0., 0., 0, 0, time.time()
        for X, y in train_iter:
            # use cpu or gpu
            X = X.to(device)
            y = y.to(device)
            # 前向传播
            y_hat = net(X)
            # 计算损失
            l = loss(y_hat, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向求导
            l.backward()
            # 一次优化
            optimizer.step()
            # 损失累加
            train_loss_sum += l.cpu().item()
            # 准确数累加
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            # 样本数累加
            n += y.shape[0]
            # 批量数+1
            batch_count += 1
        # 在验证集上验证获取验证准确度
        val_acc = evaluate_accuracy(val_iter, net)
        # 迭代轮数,训练平均损失(批量平均),训练准确度(样本平均),验证准确度(验证样本平均),一次迭代耗时
        print('epoch %d, loss %.4f, train acc %.2f, val acc %.2f, time %.1f sec'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, val_acc, time.time() - start))


def model_save(net, optimizer, vocab, category,
               batch_size, embed_size, num_channels, kernel_sizes, num_epochs, lr,
               name):
    """
    保存模型
    :param net: TextCNN模型
    :param optimizer: Adam优化器
    :param vocab: 字典
    :param category: 类别词典
    :param batch_size: 批量大小
    :param embed_size: 词向量维度
    :param num_channels: 输入通道大小
    :param kernel_sizes: 卷积核宽度
    :param num_epochs: 迭代轮数
    :param lr: 学习率
    :param name: 保存文件名
    """
    # 保存的根目录
    root = './model'
    if not os.path.exists(root):
        os.makedirs(root)
    # 保存路径
    path = root + '/' + name + '.pt'
    # 保存为字典格式
    torch.save({'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'vocab': vocab,
                'category': category,
                'batch_size': batch_size,
                'embed_size': embed_size,
                'num_channels': num_channels,
                'kernel_sizes': kernel_sizes,
                'num_epochs': num_epochs,
                'lr': lr},
               path)
    # 打印保存信息
    print('model saved as %s' % path)


def model_load(name):
    """
    加载模型
    :param name: 模型文件名
    :return: 模型,字典,类别词典
    """
    # 保存的根目录
    root = './model'
    # 输入文件名
    name += '.pt'
    # 用户输入错误
    if name not in os.listdir(root):
        print('model目录下无该模型')
        return None

    # 加载模型
    model = torch.load(root + '/' + name)
    if model is not None:
        # 载入字典和类别词典
        vocab = model['vocab']
        category = model['category']
        # 实例化网络
        net = TextCNN(len(vocab), model['embed_size'], model['num_channels'], model['kernel_sizes'])
        # 参数状态还原
        net.load_state_dict(model['model'])
    else:
        # 加载失败
        print('model load failed')
        return None

    dic = {
        'net': net,
        'vocab': vocab,
        'category': category,
    }
    return dic


def start_train(batch_size, embed_size, num_channels, kernel_sizes, num_epochs, lr):
    """
    训练入口
    :param batch_size: 批量大小
    :param embed_size: 词向量维度
    :param num_channels: 输入通道大小
    :param kernel_sizes: 卷积核宽度
    :param num_epochs: 迭代轮数
    :param lr: 学习率
    """
    # 有英伟达GPU则使用gpu,否则cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载字典,类别
    vocab, category = load_vocab(), load_category()
    # 加载数据集，验证集，测试集
    train_data, val_data, test_data = load_data()

    # 数据处理为可输入的形式
    train_feature, train_label = preprocess_data(train_data, vocab, category)
    train_iter = load_batch_iter(train_feature, train_label, batch_size)

    val_feature, val_label = preprocess_data(val_data, vocab, category)
    val_iter = load_batch_iter(val_feature, val_label, batch_size, shuffle=False)

    # 模型实例
    net = TextCNN(len(vocab), embed_size, num_channels, kernel_sizes)

    # 交叉熵损失函数
    loss = nn.CrossEntropyLoss()
    # Adam优化
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)

    # 训练
    train(train_iter, val_iter, net, loss, optimizer, device, num_epochs)

    # 是否进入测试
    user_input = raw_input('是否输入进行测试[y/n]:')
    if user_input in ['y', 'yes', '']:
        # 读取用户输入
        sentence = raw_input('输入测试语句[exit or 回车退出]:')
        # 输入非空
        while sentence != '':
            if sentence == 'exit':
                break
            # 获取类别
            label = get_label(net, vocab, category, sentence)
            # 打印
            print(label)
            # 继续监听输入
            sentence = raw_input('输入测试语句[exit or 回车退出]:')

    # 是否保存模型
    user_input = raw_input('是否保存模型[y/n]:')
    if user_input in ['y', 'yes', '']:
        # 请求保存文件名
        name = raw_input('输入模型名(不带后缀):')
        while name == '':
            name = raw_input('重新输入:')
        # 保存模型
        model_save(net, optimizer, vocab, category,
                   batch_size, embed_size, num_channels, kernel_sizes, num_epochs, lr,
                   name)


def into_train():
    """
    选择训练
    """
    print('设置训练参数:')
    # 批量大小
    batch_size = raw_input('批量大小(default:256):')
    if batch_size == '':
        batch_size = 256
    else:
        batch_size = int(batch_size)
    # 词向量维度
    embed_size = raw_input('词向量维度(default:50):')
    if embed_size == '':
        embed_size = 50
    else:
        embed_size = int(embed_size)
    # 输出通道数
    num_channels = raw_input('卷积层输出通道数(default:[30, 30,30,30]):')
    if num_channels == '':
        num_channels = [30, 30, 30, 30]
    else:
        num_channels = list(map(int, num_channels.strip('[').strip(']').split(',')))
    # 卷积核宽度
    kernel_sizes = raw_input('卷积核宽度(与通道数目保持相同,default:[1,2,3,4]):')
    if kernel_sizes == '':
        kernel_sizes = [1, 2, 3, 4]
    else:
        kernel_sizes = list(map(int, kernel_sizes.strip('[').strip(']').split(',')))
    # 迭代轮数
    num_epochs = raw_input('迭代轮数(default:5):')
    if num_epochs == '':
        num_epochs = 5
    else:
        num_epochs = int(num_epochs)
    # 学习率
    lr = raw_input('学习率(default:0.01):')
    if lr == '':
        lr = 0.01
    else:
        lr = float(lr)

    # 开始训练
    start_train(batch_size, embed_size, num_channels, kernel_sizes, num_epochs, lr)


def get_label(net, vocab, category, data):
    """
    对输入的句子判明类别
    :param net: 模型
    :param vocab: 字典
    :param category: 类别词典
    :param data: 输入的数据
    :return:
    """

    # 构造输入
    data = [[list(category.keys())[0], data]]
    # 特征,shape:(1,句子长度length)
    feature, _ = preprocess_data(data, vocab, category)
    # 送入网络,返回标签
    return {v: k for k, v in category.items()}[net(feature).argmax(dim=1).item()]


def into_test(name):
    """
    根据用户输入句子返回类别
    """
    # 载入模型,默认载入init
    name = name if name != '' else 'default'
    model = model_load(name)

    # 读取用户输入
    user_input = raw_input('请输入[exit or 回车退出]:')
    # 输入非空
    while user_input != '':
        if user_input == 'exit':
            break
        # 获取类别
        label = get_label(model['net'], model['vocab'], model['category'], user_input)
        # 打印
        print(label)
        # 继续监听输入
        user_input = raw_input('请输入[exit or 回车退出]:')


def ui():
    """
    与用户交互
    """
    welcome = 'train or test[exit or 回车退出]:'
    user_input = raw_input(welcome)
    while user_input != '':
        # 退出
        if user_input == 'exit':
            break
        # 选择训练模型
        if user_input == 'train':
            into_train()
        # 选择功能测试
        if user_input == 'test':
            into_test(raw_input('输入要加载的模型名,默认加载default.pt:'))
        user_input = raw_input(welcome)
