import torch
import random
import numpy as np

class TensorDataset(torch.utils.data.Dataset) :
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_arbitrary_masking(N, ind) : 
    indice = np.arange(0,N)
    mask = np.zeros(N, dtype=bool)
    mask[ind] = True
    return indice[~mask], indice[mask]

def k_fold_index(N = 150, k = 10, randomize = True, SEED = 10) : 
    indice = np.arange(0,N)
    if randomize is True : 
        np.random.seed(SEED)
        np.random.shuffle(indice)
    result = []
    for fold in np.split(indice, k) : 
        result.append(make_arbitrary_masking(N, fold))
    return result


def mae(predictedGaze, groundtruthGaze, is_3d = True, deg=False) :
    '''
    predictedGaze   : [N x 3] torch array of predicted gaze angle vectors with the Euclidean coordinate system
    groundtruthGaze : [N x 3] torch array of predicted gaze angle vectors with the Euclidean coordinate system

    For given predicted gaze vectors $\widehat{\bm g}_i$ and
    groundtruch gaze vectors $\bm g_i$ for i=1,2,...,n,
    compute the angles between two vectors in degree and
    return mean angular errors.
    '''

    Gaze_1 = predictedGaze.clone()
    Gaze_2 = groundtruthGaze.clone()

    if is_3d is not True : 
        Gaze_1 = convert_to_xyz(Gaze_1, deg)
        Gaze_2 = convert_to_xyz(Gaze_2, deg)

    # To avoid a numerical issues, we normalize gaze vectors again
    Gaze_1 = Gaze_1 / torch.norm(Gaze_1, dim = 1).unsqueeze(1)
    Gaze_2 = Gaze_2 / torch.norm(Gaze_2, dim = 1).unsqueeze(1)

    cos_val = torch.sum(Gaze_1 * Gaze_2, dim = 1)
    cos_val[torch.where(cos_val > 1)[0]] = 1
    cos_val[torch.where(cos_val < -1)[0]] = -1

    angle_val = torch.arccos(cos_val) * 180 / torch.pi

    return torch.mean(angle_val)


def convert_to_xyz(spherical, deg = False) :
    '''
    spherical : [N x 2] torch array of unit vectors in terms of spherical coordinates [phi, theta]

    Convert spherical coordinates to Euclidean standard coordinates.
    We follow the Xiong et al's notations,
    (\phi, \theta) -> (-\cos\phi \sin\theta, -\sin\phi, -\cos\phi \cos\theta))
    '''

    if deg is True : 
        spherical = spherical * torch.pi / 180
    xyz = torch.zeros((spherical.shape[0], 3), device=spherical.device)
    xyz[:, 0] = -torch.cos(spherical[:, 0]) * torch.sin(spherical[:, 1])
    xyz[:, 1] = -torch.sin(spherical[:, 0])
    xyz[:, 2] = -torch.cos(spherical[:, 0]) * torch.cos(spherical[:, 1])
    xyz /= torch.norm(xyz, dim = 1).unsqueeze(1)

    return xyz

def convert_to_spherical(xyz, deg = False) :
    '''
    coord : [N x 3] torch array of unit vectors in terms of Euclidean coordinates [x, y, z]

    Convert Euclidean standard coordinates to spherical coordinates.
    We follow the Xiong et al's notations,
    (\phi, \theta) <- (-\cos\phi \sin\theta, -\sin\phi, -\cos\phi \cos\theta))
    '''

    xyz /= torch.norm(xyz, dim=1).unsqueeze(1)
    spherical = torch.zeros((xyz.shape[0], 2), device=xyz.device)
    spherical[:, 0] = -torch.arcsin(xyz[:,1])
    spherical[:, 1] = torch.arctan(xyz[:,0] / xyz[:,2])

    if deg is True : 
        return spherical * 180 / torch.pi
    else : 
        return spherical
