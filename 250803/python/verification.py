# # ***************************************************************** #
# #  mem_path   : mem파일 저장된 위치
# # ***************************************************************** #

import torch
import numpy as np
import os
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import ImageOps
# ********************* a. CNN 모델 만들기 ************************ #
# *************************************************************** #

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, ch1, ch2):
        super(CNN, self).__init__()

        # pixel 데이터 추가
        self.conv1 = nn.Conv2d(1, ch1, kernel_size=5, padding=0, bias=True)
        # 합성곱 레이어

        self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max pooling
        # MaxPool2d(2, 2): 2x2 최대 풀링 → 크기를 절반으로 줄임

        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=5, padding=0, bias=True)

        self.fc1 = nn.Linear(ch2 * 4 * 4, 3, bias=True)  # 완전연결,  # a,b,c 분류
        # self.fc2 = nn.Linear(32, 3)  # a,b,c 분류

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = (x.to(torch.int32) >> t).to(torch.float32)  # <-- shift t 적용
        x = self.pool(F.relu(x))

        # conv2
        x = self.conv2(x)
        x = (x.to(torch.int32) >> k).to(torch.float32)  # <-- shift k 적용
        x = self.pool(F.relu(x))

        # flatten
        x = x.view(x.size(0), -1)

        # fc1 (shift는 안 해도 됨. 이미 bias 쪽에서 total_shift 적용됨)
        x = self.fc1(x)

        return x
# forward: 입력이 모델을 통과할 때의 연산 정의
# ReLU: 비선형 활성화 함수 → 딥러닝에서 매우 중요
# view: 텐서를 펼쳐서 FC 레이어에 넣음
# *************************************************************** #
# *************************************************************** #



# EMNIST로 정확도 평가하기 위함
transform = transforms.Compose([
    transforms.Lambda(lambda img: ImageOps.invert(img)),  # 색 반전
    transforms.Lambda(lambda img: ImageOps.mirror(img)),  # 좌우 대칭 복원
    transforms.Lambda(lambda img: img.rotate(90, expand=True)),  # 시계방향 90도 회전
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 255).round().to(torch.uint8)),
    transforms.Lambda(lambda t: t.to(torch.float32))  # ✅ 다시 float으로 바꿔줌
    ])



# # ************** a-1. EMNIST 전체 훈련/테스트 데이터셋 ************** #
# # *************************************************************** #
#
# # EMNIST 데이터셋 불러오기 (예: 'letters')
# train_dataset = EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
# test_dataset = EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
# # *************************************************************** #
# # *************************************************************** #

# # **************** a-2. A,B,C만  훈련/테스트 데이터셋 *************** #
# # *************************************************************** #
target_labels = [1, 2, 3]
def filter_dataset(dataset):
    indices = [i for i, (_, label) in enumerate(dataset) if label in target_labels]
    filtered = Subset(dataset, indices)
    return filtered

# 원본 전체 EMNIST 데이터셋
full_train_dataset = EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
full_test_dataset = EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

# 'a', 'b', 'c'만 필터링
train_dataset = filter_dataset(full_train_dataset)
test_dataset = filter_dataset(full_test_dataset)

# Subset은 내부에 index만 들고 있어서 custom Dataset 감싸줘야 함
class ABCDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return img, torch.tensor(label, dtype=torch.long)

train_dataset = ABCDataset(train_dataset)
test_dataset = ABCDataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# *************************************************************** #
# *************************************************************** #


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def evaluate(model, dataloader):
    model.eval()  # evaluation mode (dropout, batchnorm 등 off)
    correct = 0
    total = 0

    with torch.no_grad():  # gradient 계산 생략 (속도 ↑)
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)           # forward pass
            _, predicted = torch.max(outputs, 1)  # 가장 큰 값의 index
            correct += (predicted == labels - 1).sum().item()  # label: 1=A, 2=B, 3=C → 0~2로 변환
            total += labels.size(0)

    print(f"Total Test Samples: {total}")
    return correct / total * 100



# ********************* b-1. weight, bias 집어넣기 ************************ #
# *************************##******************************************** #

t = 12 #conv1후 얼마나 shift할지
k = 2 #conv2후 얼마나 shift할지
total_shift = t+k

model = CNN(3,3)

# Conv1 weights
model.conv1.weight.data = torch.tensor([
    [[[12,13,33,20,42],[11,24,54,33,28],[45,51,19,8,22],[-14,14,-22,-20,-8],[-37,-21,-22,-42,-50]]],
    [[[23,21,-11,25,26],[27,4,16,-1,26],[2,20,-3,9,61],[44,8,35,31,15],[34,47,43,31,46]]],
    [[[1,0,-4,39,23],[0,9,33,47,51],[4,-15,20,41,-1],[-49,-37,17,-16,-7],[-52,-16,-48,-32,-19]]]
], dtype=torch.float32)
# Conv2 weights
model.conv2.weight.data = torch.tensor([
    [   # Output Channel 0
       [[ 33,    9,    9, - 7 ,    0],
        [29,   22, - 13,   26,    34],
        [15,    5,   26,   38,    46],
        [19,   27,   31,   37,    33],
       [ - 2,   11,    8, - 5 , - 3]],

        [[14,   16,   14,   18,   34],
        [15, - 5 ,  15 ,  26 ,  15],
        [- 31,   13, - 1,    1,   17],
        [- 19, - 9 ,- 24, - 15, - 2],
        [- 29, - 31, - 5, - 22, - 18]],

       [[ 4 ,- 17 ,   4 ,- 8   ,- 7],
       [ 35, - 28, - 36, - 16 ,  25],
       [ 30,   27,   14,   39 ,  44],
       [ 47,   41,   45,   37 ,  31],
       [ 46,   52,   28, - 4, - 5]],
    ],

    [   # Output Channel 1
        [[ -16,  -29,  - 37, - 6 ,- 24],
        [- 27, - 27,  - 2 ,   4 ,   5],
        [- 9 , - 4 , - 1  , - 7 ,  17],
       [  11 ,   7 , - 20 ,- 6  ,  4],
       [  25 ,   7 ,   15 ,- 5  , 16]],

        [[-1 ,- 30, - 10,   22,   10],
        [13 ,  11,  - 6,    1,   38],
        [34 ,  12,    6,    7,   30],
        [33 ,  26,   30,    9,   23],
        [51 ,  25,   29,   16,    9]],

        [[-28 ,   6 , - 8 ,- 20, - 53],
       [ - 34, - 20,   3 ,- 4 ,- 47],
       [ - 45, - 40, - 8 ,  6 ,   5],
       [ - 66, - 56, -52 ,-35 ,- 28],
       [ - 41, - 45, -41 ,-47 ,- 11]],
    ],

    [   # Output Channel 2
       [[ -9,    5,    8 ,  13 ,- 10],
       [ - 8, - 16, - 2  , 10  ,  3],
       [ - 4,   11,    3 ,- 12 ,  7],
       [   1, - 13,    1 ,   1 ,- 1],
        [-10, - 14,  - 15,    0, - 6]],

       [[  11 ,  11 ,- 7, - 13 ,   4],
        [- 15 ,   6 ,- 15 ,- 9 ,   6],
        [- 9  , - 8 ,   0 ,- 1 ,- 10],
        [- 9  ,   3 ,   1 ,   2, - 9],
        [- 2  , - 4 , - 3 ,   2, -15]],

       [[ 5  ,  2 ,   3, - 15, - 9],
      [ -13 ,   2,   2, - 7 ,- 9],
       [ 1  ,  7 , - 4, - 1 ,- 15],
        [10 ,- 3 ,  11, - 13,   10],
       [ -7 ,   1,   9, - 12, - 5]],
    ]
], dtype=torch.float32)

model.fc1.weight.data = torch.tensor([
    [5, -11, -15, 26, 31, 3, -18, 1, 28, -16, -20, 5, 15, -18, -32, 3, 1, -10, 10, 6, 1, 20, -19, -12, -18, 3, -10, -27, 37, 11, 27, -12, 13, -2, -4, 5, -14, 13, -15, -6, 2, 2, -14, -17, 5, -6, 5, 12],
    [-20, 17, 26, 2, -11, -10, 24, 23, 0, -25, 5, -18, 9, 9, 15, -17, 11, 3, 15, -2, 20, -17, -15, 0, 27, 3, -24, 7, -21, -12, -19, -22, 8, 17, -12, 8, -15, 6, 3, 5, 9, 5, -11, 13, -5, -14, 5, -14],
    [20, -3, -14, -10, -20, -8, -10, -18, -7, 21, 17, 13, -24, -18, -10, 5, 0, -3, 11, -1, -33, 13, 29, 22, -3, 1, 16, 23, -5, -18, 11, -14, 6, -3, -12, 7, -8, -2, 3, -13, 15, 2, -6, -14, -10, -17, -5, 16],
], dtype=torch.float32)

# conv1 bias
model.conv1.bias.data = torch.tensor([25*256, -41*256, 43*256], dtype=torch.float32)

# # conv2 bias , need (8 + (log2(scale) + 1)-t )bit
# 1. Bitshift t만큼 한 결과를 float으로 저장 (PyTorch는 float만 허용)
raw_bias_ints = [2 * 256 * 128, -23 * 256 * 128, 1 * 256 * 128]  # bias 원본 정수
bias_shifted = [float(bias_val >> t) for bias_val in raw_bias_ints]
model.conv2.bias.data = torch.tensor(bias_shifted, dtype=torch.float32)
# 2. model.conv2.bias.data = torch.tensor([2*256*128/(2**t), -23*256*128/(2**t), 1*256*128/(2**t)], dtype=torch.float32)  # 또는 정수 bias
# 3. verilog에 집어넣어져있는 값(bias 잘못된 값)
# model.conv2.bias.data = torch.tensor([2*256, -23*256, 1*256], dtype=torch.float32)  # 또는 정수 bias

# # fc1 bias , need (8 + 2*(log2(scale) + 1) - total_shift )bit
# 1. Bitshift t만큼 한 결과를 float으로 저장 (PyTorch는 float만 허용)
raw_bias_ints = [11*256*128*128, -7*256*128*128, -18*256*128*128]
bias_shifted = [float(bias_val >> total_shift) for bias_val in raw_bias_ints]
model.fc1.bias.data = torch.tensor(bias_shifted, dtype=torch.float32)
# 2. model.fc1.bias.data = torch.tensor([11*256*128*128/((2**t)*(2**k)), -7*256*128*128/((2**t)*(2**k)), -18*256*128*128/((2**t)*(2**k))], dtype=torch.float32)
# 3. verilog에 집어넣어져있는 값(bias 잘못된 값)
# model.fc1.bias.data = torch.tensor([11*256, -7*256, -18*256], dtype=torch.float32)



# #********************* b-1. weight, bias 원본넣기 ************************ #
# #*************************##******************************************** #
# # Conv1 weights and bias
# model.conv1.weight.data = torch.tensor([[[[ 0.0942,  0.1013,  0.2582,  0.1578,  0.3310],
#           [ 0.0831,  0.1867,  0.4250,  0.2605,  0.2206],
#           [ 0.3503,  0.3999,  0.1475,  0.0629,  0.1716],
#           [-0.1133,  0.1103, -0.1731, -0.1536, -0.0653],
#           [-0.2919, -0.1652, -0.1731, -0.3274, -0.3908]]],
#         [[[ 0.1765,  0.1642, -0.0847,  0.1941,  0.1996],
#           [ 0.2124,  0.0348,  0.1257, -0.0047,  0.2043],
#           [ 0.0166,  0.1540, -0.0212,  0.0691,  0.4734],
#           [ 0.3450,  0.0665,  0.2762,  0.2388,  0.1174],
#           [ 0.2636,  0.3654,  0.3368,  0.2407,  0.3618]]],
#         [[[ 0.0068, -0.0040, -0.0287,  0.3078,  0.1771],
#           [-0.0006,  0.0665,  0.2595,  0.3644,  0.3997],
#           [ 0.0291, -0.1170,  0.1543,  0.3205, -0.0072],
#           [-0.3862, -0.2884,  0.1361, -0.1214, -0.0537],
#           [-0.4037, -0.1266, -0.3757, -0.2490, -0.1512]]]], dtype=torch.float32)
# model.conv1.bias.data = torch.tensor([ 0.1981, -0.3192,  0.3350], dtype=torch.float32)
#
# model.conv2.weight.data = torch.tensor([[[[ 0.2592,  0.0712,  0.0715, -0.0580,  0.0023],
#           [ 0.2311,  0.1724, -0.1054,  0.2066,  0.2696],
#           [ 0.1158,  0.0411,  0.2065,  0.3031,  0.3592],
#           [ 0.1499,  0.2090,  0.2459,  0.2897,  0.2585],
#           [-0.0133,  0.0900,  0.0635, -0.0385, -0.0221]],
#          [[ 0.1137,  0.1231,  0.1111,  0.1404,  0.2687],
#           [ 0.1210, -0.0417,  0.1161,  0.2051,  0.1154],
#           [-0.2439,  0.1009, -0.0055,  0.0110,  0.1335],
#           [-0.1489, -0.0715, -0.1878, -0.1183, -0.0183],
#           [-0.2312, -0.2451, -0.0377, -0.1718, -0.1398]],
#          [[ 0.0295, -0.1335,  0.0305, -0.0602, -0.0554],
#           [ 0.2743, -0.2187, -0.2796, -0.1247,  0.1965],
#           [ 0.2387,  0.2164,  0.1097,  0.3045,  0.3437],
#           [ 0.3677,  0.3251,  0.3557,  0.2930,  0.2443],
#           [ 0.3595,  0.4071,  0.2222, -0.0350, -0.0430]]],
#         [[[-0.1237, -0.2287, -0.2918, -0.0458, -0.1907],
#           [-0.2162, -0.2087, -0.0187,  0.0290,  0.0375],
#           [-0.0691, -0.0302, -0.0080, -0.0522,  0.1312],
#           [ 0.0865,  0.0529, -0.1573, -0.0454,  0.0332],
#           [ 0.1930,  0.0527,  0.1166, -0.0421,  0.1244]],
#          [[-0.0041, -0.2357, -0.0787,  0.1704,  0.0813],
#           [ 0.0986,  0.0840, -0.0510,  0.0084,  0.2955],
#           [ 0.2652,  0.0935,  0.0479,  0.0579,  0.2364],
#           [ 0.2606,  0.2075,  0.2360,  0.0723,  0.1834],
#           [ 0.4022,  0.1996,  0.2317,  0.1236,  0.0708]],
#          [[-0.2238,  0.0464, -0.0619, -0.1580, -0.4147],
#           [-0.2716, -0.1588,  0.0247, -0.0309, -0.3717],
#           [-0.3569, -0.3113, -0.0601,  0.0496,  0.0399],
#           [-0.5184, -0.4376, -0.4133, -0.2760, -0.2177],
#           [-0.3189, -0.3563, -0.3204, -0.3664, -0.0879]]],
#         [[[-0.0726,  0.0422,  0.0626,  0.1008, -0.0765],
#           [-0.0648, -0.1228, -0.0185,  0.0808,  0.0234],
#           [-0.0309,  0.0880,  0.0239, -0.0979,  0.0564],
#           [ 0.0055, -0.1021,  0.0080,  0.0100, -0.0107],
#           [-0.0768, -0.1091, -0.1176,  0.0032, -0.0478]],
#          [[ 0.0858,  0.0830, -0.0584, -0.0989,  0.0354],
#           [-0.1199,  0.0478, -0.1170, -0.0730,  0.0468],
#           [-0.0674, -0.0596, -0.0013, -0.0062, -0.0791],
#           [-0.0726,  0.0252,  0.0043,  0.0175, -0.0742],
#           [-0.0150, -0.0348, -0.0256,  0.0174, -0.1215]],
#          [[ 0.0413,  0.0130,  0.0275, -0.1187, -0.0725],
#           [-0.1053,  0.0121,  0.0139, -0.0512, -0.0700],
#           [ 0.0062,  0.0559, -0.0301, -0.0066, -0.1207],
#           [ 0.0778, -0.0234,  0.0898, -0.1014,  0.0780],
#           [-0.0589,  0.0113,  0.0741, -0.0934, -0.0371]]]], dtype=torch.float32)
# model.conv2.bias.data = torch.tensor([ 0.0135, -0.1800,  0.0046], dtype=torch.float32)  # 또는 정수 bias
#
# model.fc1.weight.data = torch.tensor([[ 0.0358, -0.0892, -0.1154,  0.2058,  0.2404,  0.0259, -0.1440,  0.0059,
#           0.2153, -0.1251, -0.1559,  0.0405,  0.1184, -0.1408, -0.2498,  0.0234,
#           0.0066, -0.0768,  0.0770,  0.0476,  0.0091,  0.1527, -0.1512, -0.0963,
#          -0.1414,  0.0247, -0.0792, -0.2076,  0.2913,  0.0852,  0.2078, -0.0901,
#           0.1050, -0.0120, -0.0286,  0.0388, -0.1076,  0.1021, -0.1197, -0.0495,
#           0.0179,  0.0120, -0.1090, -0.1356,  0.0408, -0.0453,  0.0429,  0.0946],
#         [-0.1567,  0.1298,  0.2034,  0.0122, -0.0825, -0.0784,  0.1906,  0.1770,
#          -0.0039, -0.1989,  0.0416, -0.1426,  0.0703,  0.0728,  0.1167, -0.1333,
#           0.0852,  0.0266,  0.1161, -0.0150,  0.1577, -0.1318, -0.1142, -0.0008,
#           0.2083,  0.0231, -0.1867,  0.0559, -0.1664, -0.0917, -0.1456, -0.1681,
#           0.0595,  0.1351, -0.0901,  0.0638, -0.1194,  0.0442,  0.0220,  0.0419,
#           0.0706,  0.0397, -0.0872,  0.1021, -0.0406, -0.1114,  0.0385, -0.1103],
#         [ 0.1582, -0.0269, -0.1086, -0.0754, -0.1537, -0.0654, -0.0743, -0.1402,
#          -0.0544,  0.1673,  0.1356,  0.0997, -0.1909, -0.1407, -0.0803,  0.0386,
#           0.0003, -0.0249,  0.0890, -0.0099, -0.2609,  0.1044,  0.2266,  0.1749,
#          -0.0273,  0.0117,  0.1234,  0.1758, -0.0406, -0.1432,  0.0826, -0.1058,
#           0.0456, -0.0198, -0.0950,  0.0574, -0.0639, -0.0173,  0.0196, -0.1012,
#           0.1156,  0.0139, -0.0454, -0.1070, -0.0792, -0.1303, -0.0382,  0.1213]], dtype=torch.float32)
# model.fc1.bias.data = torch.tensor([ 0.0833, -0.0529, -0.1157], dtype=torch.float32)


def load_mem_as_image(mem_path, height=28, width=28, channels=1):
    with open(mem_path, "r") as f:
        lines = f.readlines()

    values = [int(line.strip(),16) for line in lines if line.strip()]
    expected = height * width * channels

    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)}")

    arr = np.array(values, dtype=np.int16).reshape(channels, height, width)
    return torch.from_numpy(arr).float().unsqueeze(0)  # shape: [1, 1, 28, 28]

import torch
torch.set_printoptions(sci_mode=False, linewidth=150)



# *********************** image 검증용 transform ************************* #
# *************************##******************************************** #
# transform = transforms.Compose([
#     # transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# root_folder = r'C:\github\Braille_generator_FPGA\handwritebold'
# for letter in ['a', 'b', 'c']:
#     for i in range(1, 5):  # 1 to 4
#         alphabet = f'{letter}_{i}'
#         image_path = os.path.join(root_folder, f'{alphabet}.png')
#
#         try:
#             img = Image.open(image_path).convert("L")
#             img_tensor = transform(img).unsqueeze(0)





# *********************** c. mem파일 연산 결과   ************************** #
# *************************##******************************************** #
root_folder = r'C:\github\Braille_generator_FPGA\mem_outputs_dongeun'
index_to_letter = ['a', 'b', 'c']
# for letter in ['a', 'b', 'c']:
for letter in ['a', 'b', 'c']:
    for i in range(1, 6):  # 1 to 4
        alphabet = f'{letter}_{i}'
        mem_path = os.path.join(root_folder, f'{alphabet}_gray.mem')

        try:
            # ✅ 새로운 input tensor 생성
            # input_tensor = create_cyclic_input(28, 28, 1)
            input_tensor_mem = load_mem_as_image(mem_path)  # shape: [1, 1, 28, 28]


            # print("==============================================================")
            # print("🔹 새로운 Input Feature Map 정보:")
            # print(f"Shape: {input_tensor_mem.shape}")
            # print(f"Min value: {input_tensor_mem.min().item()}")
            # print(f"Max value: {input_tensor_mem.max().item()}")



            # 일부 값들 확인 (좌상단 5x5 영역)
            # print("\n Input Feature Map 좌상단 5x5 영역:")
            # print(input_tensor_mem[0, 0, :29, :29])

            print(f"\n📄 Input mem 연산: {alphabet}")
            # # ✅ 이제 이 input_tensor로 연산 수행
            # print("\n" + "="*80)
            # print("🔹  Input으로 연산 시작")
            # print("="*80)


            #
            # ✅ Conv1 연산
            conv1_out = model.conv1(input_tensor_mem)
            # print("🔹 [Conv1 출력] shape:", conv1_out.shape)
            # for i in range(conv1_out.shape[1]):
            #     print(f"Conv1 채널 {i} 값:")
            #     print(conv1_out[0, i])

            # float → int 변환 (예: 32비트 정수)
            conv1_int = conv1_out.to(torch.int32)

            # 8비트 오른쪽 shift
            conv1_shifted = conv1_int >> t

            # (선택) float로 다시 변환
            conv1_shifted = conv1_shifted.to(torch.float32)

            # ✅ ReLU 적용
            relu1_out = F.relu(conv1_shifted)
            # print("\n🔹 [ReLU1 출력] (Conv1 → ReLU)")
            # for i in range(relu1_out.shape[1]):
            #     print(f"ReLU 채널 {i} 값:")
            #     print(relu1_out[0, i])

            # ✅ MaxPooling1 적용
            pool1_out = F.max_pool2d(relu1_out, 2, 2)
            # print("\n🔹 [MaxPool1 출력] shape:", pool1_out.shape)
            # for i in range(pool1_out.shape[1]):
            #     print(f"MaxPool1 채널 {i} 값:")
            #     print(pool1_out[0, i])




            # ✅ Conv2 연산
            conv2_out = model.conv2(pool1_out)
            # print("\n🔹 [Conv2 출력] shape:", conv2_out.shape)
            # for i in range(conv2_out.shape[1]):
            #     print(f"Conv2 채널 {i} 값:")
            #     print(conv2_out[0, i])

            # float → int 변환 (예: 32비트 정수)
            conv2_int = conv2_out.to(torch.int32)

            # 8비트 오른쪽 shift
            conv2_shifted = conv2_int >> k

            # (선택) float로 다시 변환
            conv2_shifted = conv2_shifted.to(torch.float32)


            # ✅ ReLU2 적용
            relu2_out = F.relu(conv2_out)
            # print("\n🔹 [ReLU2 출력] (Conv2 → ReLU)")
            # for i in range(relu2_out.shape[1]):
            #     print(f"ReLU2 채널 {i} 값:")
            #     print(relu2_out[0, i])

            # ✅ MaxPooling2 적용
            pool2_out = F.max_pool2d(relu2_out, 2, 2)
            # print("\n🔹 [MaxPool2 출력] shape:", pool2_out.shape)
            # for i in range(pool2_out.shape[1]):
            #     print(f"MaxPool2 채널 {i} 값:")
            #     print(pool2_out[0, i])

            # ✅ Flatten (벡터화)
            flatten_out = pool2_out.view(pool2_out.size(0), -1)
            # print("\n🔹 [Flatten 출력] shape:", flatten_out.shape)
            # print("Flatten 값:")
            # print(flatten_out[0])  # 배치 크기 1이므로 [0]만 출력


            # ✅ FC1 연산 (Fully Connected Layer)
            fc1_out = model.fc1(flatten_out)


            # float → int 변환 (예: 32비트 정수)
            fc1_int = fc1_out.to(torch.int32)

            # 8비트 오른쪽 shift
            fc1_shifted = fc1_int >> 0

            # (선택) float로 다시 변환
            fc1_shifted = fc1_shifted.to(torch.float32)

            # ✅ 예측 클래스 인덱스 → 알파벳으로 변환
            pred_index = torch.argmax(fc1_shifted[0]).item()
            pred_letter = index_to_letter[pred_index]

            print(f"\n📄 Input mem: {alphabet}")
            print(f"🔹 FC1 Output Vector: {fc1_shifted[0].tolist()}")
            print(f"✅ Predicted Letter: '{pred_letter.upper()}' (index {pred_index})")
        except Exception as e:
            print(f"Failed to process {mem_path}: {e}")


# ⬇ 모델 GPU로 보내기
model = model.to(device)

# ⬇ 정확도 출력
accuracy = evaluate(model, test_loader)
print(f"\nTest Accuracy: {accuracy:.2f}%")

weight_bias_path = r'C:\github\Braille_generator_FPGA\test'
pixel_scale = 256
scale = 128  # 보통 -1.0 ~ 1.0 사이의 값이면 128 곱해서 int8 사용


# conv1.bias
bias = model.conv1.bias.data.clone().cpu()
print("\nconv1_bias.shape:", bias.shape)
# 정수 변환 → int16 clamp
int_bias = bias.round().to(torch.int32).clamp(-32768, 32767).to(torch.int16)
print("conv1_int_bias.shape:", int_bias.shape)
with open(os.path.join(weight_bias_path, "conv1_bias.mem"), "w") as f:
    for out_ch in range(int_bias.shape[0]):
        val = int_bias[out_ch].item()
        hex_val = f"{(val & 0xFFFF):04x}"  # 2-digit hex (8bit signed)
        f.write(f"0x{hex_val}\n")
print("conv1_bias_write_done")


# conv2.bias
bias = model.conv2.bias.data.clone().cpu()
print("\nconv2_bias.shape:", bias.shape)
int_bias = bias.round().to(torch.int32).clamp(-32768, 32767).to(torch.int16)
print("conv2_int_bias.shape:", int_bias.shape)
with open(os.path.join(weight_bias_path, "conv2_bias.mem"), "w") as f:
    for out_ch in range(int_bias.shape[0]):
        val = int_bias[out_ch].item()
        hex_val = f"{(val & 0xFFFF):04x}"  # 2-digit hex (8bit signed)
        f.write(f"0x{hex_val}\n")
print("conv2_bias_write_done")


# fc1.bias
bias = model.conv2.bias.data.clone().cpu()
print("\nfc1_bias.shape:", bias.shape)
int_bias = bias.round().to(torch.int32).clamp(-32768, 32767).to(torch.int16)
print("fc1_int_bias.shape:", int_bias.shape)
with open(os.path.join(weight_bias_path, "stage3_fc1_bias.mem"), "w") as f:
    for out_ch in range(int_bias.shape[0]):
        val = int_bias[out_ch].item()
        hex_val = f"{(val & 0xFFFF):04x}" # 2-digit hex (8bit signed)
        f.write(f"0x{hex_val}\n")
print("fc1_bias_write_done")




print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹여기서 부터는 이미지로 검증🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
# # Conv1 weights and bias
# model.conv1.weight.data = torch.tensor([[[[ 0.0942,  0.1013,  0.2582,  0.1578,  0.3310],
#           [ 0.0831,  0.1867,  0.4250,  0.2605,  0.2206],
#           [ 0.3503,  0.3999,  0.1475,  0.0629,  0.1716],
#           [-0.1133,  0.1103, -0.1731, -0.1536, -0.0653],
#           [-0.2919, -0.1652, -0.1731, -0.3274, -0.3908]]],
#         [[[ 0.1765,  0.1642, -0.0847,  0.1941,  0.1996],
#           [ 0.2124,  0.0348,  0.1257, -0.0047,  0.2043],
#           [ 0.0166,  0.1540, -0.0212,  0.0691,  0.4734],
#           [ 0.3450,  0.0665,  0.2762,  0.2388,  0.1174],
#           [ 0.2636,  0.3654,  0.3368,  0.2407,  0.3618]]],
#         [[[ 0.0068, -0.0040, -0.0287,  0.3078,  0.1771],
#           [-0.0006,  0.0665,  0.2595,  0.3644,  0.3997],
#           [ 0.0291, -0.1170,  0.1543,  0.3205, -0.0072],
#           [-0.3862, -0.2884,  0.1361, -0.1214, -0.0537],
#           [-0.4037, -0.1266, -0.3757, -0.2490, -0.1512]]]], dtype=torch.float32)
# model.conv1.bias.data = torch.tensor([ 0.1981, -0.3192,  0.3350], dtype=torch.float32)
#
#
# # Conv2 weights and bias
# model.conv2.weight.data = torch.tensor([[[[ 0.2592,  0.0712,  0.0715, -0.0580,  0.0023],
#           [ 0.2311,  0.1724, -0.1054,  0.2066,  0.2696],
#           [ 0.1158,  0.0411,  0.2065,  0.3031,  0.3592],
#           [ 0.1499,  0.2090,  0.2459,  0.2897,  0.2585],
#           [-0.0133,  0.0900,  0.0635, -0.0385, -0.0221]],
#          [[ 0.1137,  0.1231,  0.1111,  0.1404,  0.2687],
#           [ 0.1210, -0.0417,  0.1161,  0.2051,  0.1154],
#           [-0.2439,  0.1009, -0.0055,  0.0110,  0.1335],
#           [-0.1489, -0.0715, -0.1878, -0.1183, -0.0183],
#           [-0.2312, -0.2451, -0.0377, -0.1718, -0.1398]],
#          [[ 0.0295, -0.1335,  0.0305, -0.0602, -0.0554],
#           [ 0.2743, -0.2187, -0.2796, -0.1247,  0.1965],
#           [ 0.2387,  0.2164,  0.1097,  0.3045,  0.3437],
#           [ 0.3677,  0.3251,  0.3557,  0.2930,  0.2443],
#           [ 0.3595,  0.4071,  0.2222, -0.0350, -0.0430]]],
#         [[[-0.1237, -0.2287, -0.2918, -0.0458, -0.1907],
#           [-0.2162, -0.2087, -0.0187,  0.0290,  0.0375],
#           [-0.0691, -0.0302, -0.0080, -0.0522,  0.1312],
#           [ 0.0865,  0.0529, -0.1573, -0.0454,  0.0332],
#           [ 0.1930,  0.0527,  0.1166, -0.0421,  0.1244]],
#          [[-0.0041, -0.2357, -0.0787,  0.1704,  0.0813],
#           [ 0.0986,  0.0840, -0.0510,  0.0084,  0.2955],
#           [ 0.2652,  0.0935,  0.0479,  0.0579,  0.2364],
#           [ 0.2606,  0.2075,  0.2360,  0.0723,  0.1834],
#           [ 0.4022,  0.1996,  0.2317,  0.1236,  0.0708]],
#          [[-0.2238,  0.0464, -0.0619, -0.1580, -0.4147],
#           [-0.2716, -0.1588,  0.0247, -0.0309, -0.3717],
#           [-0.3569, -0.3113, -0.0601,  0.0496,  0.0399],
#           [-0.5184, -0.4376, -0.4133, -0.2760, -0.2177],
#           [-0.3189, -0.3563, -0.3204, -0.3664, -0.0879]]],
#         [[[-0.0726,  0.0422,  0.0626,  0.1008, -0.0765],
#           [-0.0648, -0.1228, -0.0185,  0.0808,  0.0234],
#           [-0.0309,  0.0880,  0.0239, -0.0979,  0.0564],
#           [ 0.0055, -0.1021,  0.0080,  0.0100, -0.0107],
#           [-0.0768, -0.1091, -0.1176,  0.0032, -0.0478]],
#          [[ 0.0858,  0.0830, -0.0584, -0.0989,  0.0354],
#           [-0.1199,  0.0478, -0.1170, -0.0730,  0.0468],
#           [-0.0674, -0.0596, -0.0013, -0.0062, -0.0791],
#           [-0.0726,  0.0252,  0.0043,  0.0175, -0.0742],
#           [-0.0150, -0.0348, -0.0256,  0.0174, -0.1215]],
#          [[ 0.0413,  0.0130,  0.0275, -0.1187, -0.0725],
#           [-0.1053,  0.0121,  0.0139, -0.0512, -0.0700],
#           [ 0.0062,  0.0559, -0.0301, -0.0066, -0.1207],
#           [ 0.0778, -0.0234,  0.0898, -0.1014,  0.0780],
#           [-0.0589,  0.0113,  0.0741, -0.0934, -0.0371]]]], dtype=torch.float32)
# model.conv2.bias.data = torch.tensor([ 0.0135, -0.1800,  0.0046], dtype=torch.float32)
#
#
# # FC1
# model.fc1.weight.data = torch.tensor([
#     [0.0358, -0.0892, -0.1154, 0.2058, 0.2404, 0.0259, -0.1440, 0.0059,
#      0.2153, -0.1251, -0.1559, 0.0405, 0.1184, -0.1408, -0.2498, 0.0234,
#      0.0066, -0.0768, 0.0770, 0.0476, 0.0091, 0.1527, -0.1512, -0.0963,
#      -0.1414, 0.0247, -0.0792, -0.2076, 0.2913, 0.0852, 0.2078, -0.0901,
#      0.1050, -0.0120, -0.0286, 0.0388, -0.1076, 0.1021, -0.1197, -0.0495,
#      0.0179, 0.0120, -0.1090, -0.1356, 0.0408, -0.0453, 0.0429, 0.0946],
#
#     [-0.1567,  0.1298,  0.2034,  0.0122, -0.0825, -0.0784,  0.1906,  0.1770,
#          -0.0039, -0.1989,  0.0416, -0.1426,  0.0703,  0.0728,  0.1167, -0.1333,
#           0.0852,  0.0266,  0.1161, -0.0150,  0.1577, -0.1318, -0.1142, -0.0008,
#           0.2083,  0.0231, -0.1867,  0.0559, -0.1664, -0.0917, -0.1456, -0.1681,
#           0.0595,  0.1351, -0.0901,  0.0638, -0.1194,  0.0442,  0.0220,  0.0419,
#           0.0706,  0.0397, -0.0872,  0.1021, -0.0406, -0.1114,  0.0385, -0.1103],
#
#     [0.1582, -0.0269, -0.1086, -0.0754, -0.1537, -0.0654, -0.0743, -0.1402,
#      -0.0544, 0.1673, 0.1356, 0.0997, -0.1909, -0.1407, -0.0803, 0.0386,
#      0.0003, -0.0249, 0.0890, -0.0099, -0.2609, 0.1044, 0.2266, 0.1749,
#      -0.0273, 0.0117, 0.1234, 0.1758, -0.0406, -0.1432, 0.0826, -0.1058,
#      0.0456, -0.0198, -0.0950, 0.0574, -0.0639, -0.0173, 0.0196, -0.1012,
#      0.1156, 0.0139, -0.0454, -0.1070, -0.0792, -0.1303, -0.0382, 0.1213]
# ], dtype=torch.float32)
#
# model.fc1.bias.data = torch.tensor([ 0.0833, -0.0529, -0.1157], dtype=torch.float32)
#
#
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
#
#
# def print_cnn_intermediate_outputs(model, input_tensor):
#     x = input_tensor.clone()
#
#     print("Input:", x.shape)
#
#     print("conv1 weights:")
#     print(model.conv1.weight.data)
#
#     x = model.conv1(x)
#     print("After conv1:", x[0, 0, :5, :5])
#
#     x = F.relu(x)
#     print("After ReLU1:", x[0, 0, :5, :5])
#
#     x = model.pool(x)
#     print("After pool1:", x[0, 0, :5, :5])
#
#     x = model.conv2(x)
#     print("After conv2:", x[0, 0, :5, :5])
#
#     x = F.relu(x)
#     print("After ReLU2:", x[0, 0, :5, :5])
#
#     x = model.pool(x)
#     print("After pool2:", x)
#
#     x = x.view(x.size(0), -1)
#     print("After flatten:", x)
#
#     x = model.fc1(x)
#     print("After fc1:", x)
#
#     return x
# transform = transforms.Compose([
#     # transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# root_folder = r'C:\github\Braille_generator_FPGA\handwritebold'
# for letter in ['a', 'b', 'c']:
#     for i in range(1, 5):  # 1 to 4
#         alphabet = f'{letter}_{i}'
#         image_path = os.path.join(root_folder, f'{alphabet}.png')
#
#         try:
#             img = Image.open(image_path).convert("L")
#             img_tensor = transform(img).unsqueeze(0)
#
#             # CNN 결과 출력
#             print(f"\n====== {alphabet}.png ======")
#             output = model(img_tensor)
#             _, predicted = torch.max(output, 1)
#             predicted_letter = chr(predicted.item() + ord('a'))
#
#             print("Output logits:", output)
#             print(f"Predicted: {predicted_letter}")
#
#             # 중간 결과 출력
#             # print_cnn_intermediate_outputs(model, img_tensor)
#
#             # 이미지 시각화
#             img_np = img_tensor.squeeze().numpy()
#             plt.imshow(img_np, cmap="gray")
#             plt.title(f"{alphabet}.png")
#             plt.axis("off")
#             plt.show()
#
#         except Exception as e:
#             print(f"Failed to process {image_path}: {e}")
# print()
