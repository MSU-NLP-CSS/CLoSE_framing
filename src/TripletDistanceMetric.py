from enum import Enum
import torch.nn.functional as F

class TripletDistanceMetric(Enum):
    '''
    The metric for the triplet loss
    '''
    COSINE = lambda x, y: 1 - F.cosine_similarity(x,y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)