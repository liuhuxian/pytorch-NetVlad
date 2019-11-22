import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

root_dir = r'/home/huxian/drive/file-drive/HUXIAN/项目数据集/netvlad/Pittsburgh250k'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)

def get_whole_val_set():
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_250k_val_set():
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())
def get_whole_test_set():
    structFile = join(struct_dir, 'pitts30k_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_250k_test_set():
    structFile = join(struct_dir, 'pitts250k_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_training_query_set(margin=0.1):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform(), margin=margin)

def get_val_query_set():
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_250k_val_query_set():
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())
'''
whichSet:train test
dataset:数据集名字
dbImage：图片地址
utmDb；dbImage图片对应的经纬度坐标
qImage:论文中query数据集的图片地址
utmQ:qImage图片对应的经纬度坐标
numDb：dbImage的图片的数量
numQ:qImage的图片的数量
posDistThr: 计算potential_negatives集合的一个阈值
posDistSqThr:没用到
nonTrivPosDistSqThr:而此参数是图片Q从dbImage在nonTrivPosDistSqThr范围内搜索得到nontrivial positive集合的一个阈值。
'''

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

# 读取数据集
def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        # 图片的地址
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            # 对self.dbStruct.utmDb里的坐标进行聚类
            knn.fit(self.dbStruct.utmDb)

            # 在self.dbStruct.utmDb中找出距离self.dbStruct.utmQ距离为self.dbStruct.posDistThr的图片，返回相应的距离和图片的坐标
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
        
def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        # 此构造函数最重要的两个功能是构造nontrivial_positives集合以及potential_negatives集合
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        # 对数据集中的经纬度进行聚类，也就是对实际的距离进行聚类，以在后面得到nontrivial_positives集合
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        # 在dbimage中，找出与qImage图片范围在self.nontrivial_positives**0.5的照片
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        # 有一些query照片可能没有对应的nontrivial_positives集合，把这些照片剔除出去
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives are those outside of posDistThr range
        # 为后面计算potential_negatives做准备。对于self.dbStruct.utmQ中的每张图片Q，从dbimage中，找出范围在self.dbStruct.posDistThr的照片
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)

        self.potential_negatives = []
        # 简单的说就是potential_negatives=dbStruct.numDb-potential_positives
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))

        # 储存着所有train_set的feature的HDF5文件的地址
        self.cache = None # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        # self.queries相当与是筛选了空的nontrivial positive后的引索表
        index = self.queries[index] # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb
            # query照片对应的netvlad特征
            qFeat = h5feat[index+qOffset]

            # nontrivial_positives对应的netvlad特征集合
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            # 在posFeat中，使用knn找出和qFeat最近的那个netvlad特征。其找出的netvlad特征储存在posNN。
            # dPos是posNN与qFeat的距离
            knn = NearestNeighbors(n_jobs=-1) # TODO replace with faiss?
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            # 找出posNN在dbImage中的引索
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            # potential_negatives数量太多，需要随机出去。以下是在self.potential_negatives[index]中随机抽取
            # 长度为self.nNegSample的序列
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            # 利用knn找出和qFeat在netvlad维度上相近的negFeat，其数量为self.nNeg*10
            # dNeg储存negNN与qFeat之间的距离
            negFeat = h5feat[negSample.tolist()]
            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), 
                    self.nNeg*10) # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            # 到这一步negNN有self.nNeg*10，还需要进行筛选，以下就是从negNN筛选出其距离不满足论文里的margin公式的。
            # 为什么这么做呢？相当于你现在有一个qFeat，posNN,和negNN集合。通常negNN集合里的特征和qFeat的距离应该都比
            # dPos大，但是会出现没有它大的情况，这些negNN对训练模型更有用
            violatingNeg = dNeg < dPos + self.margin**0.5
     
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            # 从上一步筛选的特征中，选择self.nNeg个。这里其实我觉得可以随机选择
            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = Image.open(join(queries_dir, self.dbStruct.qImage[index]))
        positive = Image.open(join(root_dir, self.dbStruct.dbImage[posIndex]))

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(root_dir, self.dbStruct.dbImage[negIndex]))
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)
