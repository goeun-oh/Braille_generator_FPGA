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

        self.fc1 = nn.Linear(ch2 * 4 * 4, 26, bias=True)  # 완전연결,  # a,b,c 분류
        # self.fc2 = nn.Linear(32, 3)  # a,b,c 분류

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = (x.to(torch.int32) >> t).to(torch.float32) # <-- shift t 적용

        # conv2
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = (x.to(torch.int32) >> k).to(torch.float32)  # <-- shift k 적용

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
k = 8 #conv2후 얼마나 shift할지
total_shift = t+k

model = CNN(3,3)

# Conv1 weights
model.conv1.weight.data = torch.tensor([
    [
        [
            [3, 30, -29, -48, -8],
            [61, 38, -76, -14, 15],
            [69, -58, -82, -8, 55],
            [-7, -80, -40, 27, 61],
            [-51, -8, 10, 43, 21],
        ],
    ],
    [
        [
            [-38, -28, 5, 27, 44],
            [-38, -38, 25, 27, 51],
            [-25, -68, -5, 41, 54],
            [-26, -64, -14, 63, 32],
            [-46, -54, -11, 65, 47],
        ],
    ],
    [
        [
            [-15, 11, 37, 58, 84],
            [12, 8, 47, 22, 75],
            [25, 43, 28, 57, 44],
            [20, 16, 45, 44, 48],
            [19, 60, 80, 54, 55],
        ],
    ],
]
, dtype=torch.float32)
# Conv2 weights
model.conv2.weight.data = torch.tensor([
    [
        [
            [-77, -40, -25, -26, -52],
            [-27, -51, -27, 16, -26],
            [-9, -16, 9, 34, -15],
            [3, 17, 20, 25, -41],
            [0, 19, -9, 23, -68],
        ],
        [
            [-55, -5, -37, -38, -89],
            [-35, -28, -43, -40, -58],
            [-48, -31, -35, -23, 5],
            [-24, -5, 6, -1, 19],
            [-21, 20, 10, -1, 5],
        ],
        [
            [23, 22, 43, 27, 10],
            [30, 25, 27, -2, -23],
            [27, 12, -6, -7, -8],
            [47, 9, 14, 5, 17],
            [-19, -60, -16, -7, 14],
        ],
    ],
    [
        [
            [2, 37, 43, 64, 36],
            [34, 71, 61, 50, -4],
            [35, 53, 33, 51, 38],
            [-3, 22, 54, 64, 69],
            [-61, -53, -36, 3, 0],
        ],
        [
            [9, -31, 0, 11, 5],
            [10, 24, 44, 29, 10],
            [18, 26, 41, 19, -12],
            [13, -15, -19, -42, -20],
            [-23, -82, -58, -72, -43],
        ],
        [
            [-36, -29, -54, -48, -30],
            [-20, -27, -6, 7, -12],
            [-3, -5, 22, 16, 23],
            [2, 9, 28, 40, 48],
            [-9, 14, 28, 48, 71],
        ],
    ],
    [
        [
            [-7, 34, 46, 52, 49],
            [46, 77, 60, 62, 42],
            [42, 83, 61, 93, 83],
            [10, 54, 65, 85, 42],
            [-37, -25, -17, 31, 39],
        ],
        [
            [54, 69, 90, 37, -31],
            [55, 59, 38, -15, -30],
            [76, 78, 13, -23, -29],
            [61, 54, 16, -1, -36],
            [-25, -50, -26, -27, -42],
        ],
        [
            [-3, 16, 8, 4, -9],
            [25, 17, -16, 1, 3],
            [15, -6, -15, 3, 0],
            [20, -18, -12, 4, 19],
            [-6, -2, -17, -16, -22],
        ],
    ],
]
, dtype=torch.float32)

model.fc1.weight.data = torch.tensor( [
    [32, 10, -12, -25, 27, -6, -15, -31, -4, 16, -18, -10, 1, 3, -7, -32, 20, 21, -17, -22, 7, 11, -19, -29, 12, 6, -9, -19, 26, 18, 3, -12, -25, 2, -25, -23, 13, -15, 11, 2, 33, -25, 2, 1, 11, 17, -3, 25],
    [0, 0, -5, 6, 11, -12, 12, -7, 22, 10, -21, 4, -9, 6, -3, -25, 9, -13, -8, -7, -24, -6, -15, 1, -13, 2, 32, -54, -40, -1, -28, 38, 21, 9, 9, -15, 19, 14, 10, 13, -17, 21, -32, -58, 27, 12, 30, 20],
    [30, 21, -4, 12, 31, 24, -27, 23, -40, -28, -8, 19, -29, -22, 6, 21, 38, 18, -4, 8, 11, 13, 7, 25, -7, -26, -4, 24, -49, -46, -9, -10, -48, -35, -11, -7, 35, 2, 16, 6, 1, 14, -6, 13, -35, -12, -49, -32],
    [2, 7, -5, 21, -3, 2, 5, -4, 25, -1, 2, -10, 1, -25, -8, -7, 23, 23, -1, -31, -8, 21, 16, -27, -5, 4, -14, -10, -22, 6, -31, 16, -55, -50, -43, 12, -35, 2, 0, -4, 9, 5, -12, 30, 35, 33, -8, 31],
    [48, 28, 19, 18, -5, -34, -38, 10, -19, -5, -6, 23, -15, -45, -4, 17, 20, 12, 22, 6, -28, 5, -1, 20, 1, -8, 29, 31, -19, -35, -26, 16, 20, 11, -28, -34, -6, -32, 4, 11, 33, -9, -21, 4, -42, -23, -61, -23],
    [16, 4, -13, -25, 4, -5, -9, -18, 4, -29, -23, 30, -4, 29, 17, 47, 25, 9, 31, 42, -29, -11, -15, -14, 5, -32, 13, 28, 20, 0, 28, 2, -10, 22, 4, -18, -15, -9, -7, -46, -37, -20, -16, -23, -26, -18, -15, -33],
    [30, 2, -22, -17, 15, 0, -14, 1, -2, 26, -6, -15, 10, -3, -9, -35, 14, 3, 1, 19, -6, 20, -3, -13, -3, 8, 15, -13, -34, 1, 4, 9, 39, 9, 9, -18, 30, 13, -18, -14, 0, -30, -23, -26, 5, 1, 3, 12],
    [-20, -3, -2, 6, 5, 12, 16, 0, -14, 8, -1, 16, -2, 32, -12, -34, -11, -30, -17, -11, -20, 6, -33, -20, -7, -11, 1, -28, 31, 32, -15, -23, 5, 31, 16, 47, -14, 21, -15, -12, -40, -8, -8, -23, 24, 40, 13, 36],
    [15, -11, 14, 23, -4, 1, -8, 11, 19, 10, -13, 26, -1, -2, -8, 13, -7, 14, 24, 4, -5, -6, 12, -10, -12, -10, 25, -32, -29, 6, 21, -25, -57, -36, -35, -1, -38, -35, -3, -8, 8, 10, 3, 1, -53, -10, 3, 2],
    [4, 3, -2, 21, 22, 8, -20, 13, 18, 24, -17, -10, -43, -9, 34, 36, 0, 7, 13, 17, 31, 19, -12, -16, -29, -7, -11, -32, -27, -21, 34, -5, -47, -57, -47, -40, -71, -34, -7, 12, 3, -16, 2, 7, -1, 25, 17, 15],
    [-29, -22, 4, 31, 28, 2, 2, 41, -23, 21, 24, 9, -7, -1, -35, -30, -21, -50, -7, 1, -38, -3, 11, 31, -14, -4, -8, -5, 20, -5, -20, -49, 37, 50, 42, 25, 21, 3, -4, -7, -30, -8, -9, -10, 22, 1, -37, -2],
    [-6, -7, 7, 21, 5, -13, 12, 6, 2, 8, 35, 37, -19, -11, 11, 39, 4, -30, 12, -19, 5, -36, 35, -23, -20, -19, 23, -28, -20, -9, 16, -32, -32, 0, 7, 22, -41, -11, -9, 2, 1, 21, 3, 11, -15, -4, -19, -3],
    [-31, 4, -7, -15, -29, -3, 23, -42, -49, -10, 18, -59, 5, 25, 19, -40, -20, -9, -8, -60, 5, -1, 3, -57, 32, 34, 29, -17, 22, 8, 17, -28, -28, -34, 2, 6, -21, 0, 16, -18, 1, 13, 29, 27, 61, -12, 21, 35],
    [-14, 0, -5, -1, -10, -4, 12, -32, -13, -2, 22, -42, -1, 7, -1, -5, -23, 4, -6, -44, 22, 15, -7, -21, 23, 1, -11, -29, 71, 5, -10, 1, -20, -15, -45, 0, -34, 6, 8, 8, -37, -26, 38, 19, 34, 34, 36, 5],
    [21, 18, -8, -11, 13, 29, -21, -20, -30, -4, -21, -29, -26, -12, -2, 19, 18, 21, 3, -28, 10, 36, 25, -34, -16, 21, 2, 12, -37, -29, 18, 41, -38, -17, -35, -17, 7, 2, -8, -17, 16, -17, -6, -17, -16, 7, -9, 28],
    [-23, -8, -24, -45, -9, 1, -10, -11, -7, -37, 17, 40, 12, 4, 17, 24, 14, 17, 15, -20, -21, -3, -10, 41, 1, -11, 13, 71, -1, -12, 16, 22, -5, -1, -13, -19, -11, -4, 25, 28, -45, -6, -7, -14, -8, -9, -21, -90],
    [25, -2, -22, -43, 12, 0, -11, -10, -3, 16, -18, -2, 24, 3, -21, -30, 3, 22, 9, -17, -3, 3, 5, -11, 22, 3, -17, 6, -14, -3, 5, -9, 54, 15, -5, -31, 19, -3, -16, 3, 36, -30, 9, 22, -12, -5, 16, -1],
    [5, 2, 8, 0, 1, -8, -15, 5, -16, -20, 10, -15, 3, -2, -34, 4, 14, 7, 23, -24, 6, 6, 20, 24, 2, 0, -16, 31, 39, 19, -19, -40, 14, 21, -34, -16, -30, 3, 0, -8, -37, -15, -6, -15, -20, 31, 4, -12],
    [48, 16, 10, 16, -3, -29, -16, 4, 43, 11, -5, 0, -12, -15, -2, -6, 6, -11, 29, 39, -4, 6, 3, -14, -44, 12, 21, -47, -56, -11, 7, 49, 72, 22, -24, -42, -66, -18, -14, -48, -9, -16, -61, -10, -27, -13, -14, -4],
    [-15, -11, 5, 1, -14, -4, -14, 23, -1, -26, -9, 16, 25, -4, -13, 14, 20, 1, 15, 6, 34, -1, 8, 5, 36, 8, 23, 19, 5, -4, 4, -2, -58, -64, 4, 19, -26, -3, 11, -24, -42, -23, -17, -51, -5, 16, 29, 2],
    [-15, 22, -3, 5, -3, 8, 18, -28, -39, 25, 19, 14, -37, 3, 26, -19, -3, -15, -32, -45, -7, -17, -28, -7, -11, 5, -30, -6, -16, 2, -1, -4, -15, 15, 36, 24, 20, 18, 19, 12, 19, 23, -6, 26, -10, 1, -11, 18],
    [-31, -19, 8, -4, -32, -14, -5, 2, -24, 2, 36, 7, -14, 7, 46, 13, -42, -29, -9, 10, -22, -29, -10, 29, -1, -13, -8, 13, 6, -7, 21, 2, -21, 0, -1, 6, -19, 26, -4, -9, 6, 24, 10, 28, -31, 6, 18, -7],
    [-29, 6, 17, -20, -11, 0, 31, -34, -42, 3, 7, -21, -11, 6, -9, 10, -42, 7, 8, -69, -39, -13, -23, -33, -7, 9, -23, 7, 11, 2, -23, 18, 5, -24, -11, 0, 44, 0, 11, 19, 28, -15, 19, 32, 22, 6, 30, 28],
    [-28, -36, 9, 27, -28, -6, -4, 28, 27, 17, 21, -6, 18, 28, -39, -31, -47, -70, 11, 18, -23, -1, -9, 28, -6, -12, 16, -32, 22, 21, -8, -39, 27, 31, 17, 14, -29, -10, -3, 13, -16, 5, -13, -6, 38, -18, -14, -19],
    [-37, -51, -14, -12, -20, 8, 19, 18, -20, 24, 26, 5, 11, 16, 37, 9, -29, -67, -14, 36, 0, 13, -17, 12, -1, 8, 9, 14, -2, -14, 27, -16, -12, 47, 21, 23, 14, 5, -4, -11, -54, -12, 3, 28, -15, 0, -8, -15],
    [-17, -23, -25, 34, -10, -6, -10, 10, 51, 14, 26, 17, 2, -21, -44, -15, 22, 16, 4, -1, 8, -4, 13, 13, 19, 13, 20, 5, -18, -1, -15, 0, -18, -56, -49, 8, -26, -14, -4, 38, 22, 32, 35, -2, -15, -22, -47, -69],
]
, dtype=torch.float32)

# conv1 bias
model.conv1.bias.data = torch.tensor([12*256, -12*256, -121*256], dtype=torch.float32)

# # conv2 bias , need (8 + (log2(scale) + 1)-t )bit
# 1. Bitshift t만큼 한 결과를 float으로 저장 (PyTorch는 float만 허용)
raw_bias_ints = [-30 * 256 * 128, -29 * 256 * 128, -57 * 256 * 128]  # bias 원본 정수
bias_shifted = [float(bias_val >> t) for bias_val in raw_bias_ints]
model.conv2.bias.data = torch.tensor(bias_shifted, dtype=torch.float32)
# 2. model.conv2.bias.data = torch.tensor([2*256*128/(2**t), -23*256*128/(2**t), 1*256*128/(2**t)], dtype=torch.float32)  # 또는 정수 bias
# 3. verilog에 집어넣어져있는 값(bias 잘못된 값)
# model.conv2.bias.data = torch.tensor([2*256, -23*256, 1*256], dtype=torch.float32)  # 또는 정수 bias

# # fc1 bias , need (8 + 2*(log2(scale) + 1) - total_shift )bit
# 1. Bitshift t만큼 한 결과를 float으로 저장 (PyTorch는 float만 허용)
# raw_bias_ints = [11*256*128*128, -7*256*128*128, -18*256*128*128]
fc1_bias = [25, 36, -1, 29, 56, 19, 1, -32, 2, 20, 5, -25, -15, -6, -3, -26, 14, 32, 23, 9, -32, -18, -9, 11, -50, 45]
# 각 요소에 대해 256 * 128 * 128을 곱하기
scale = 256 * 128 * 128
scaled_bias = [x * scale for x in fc1_bias]
bias_shifted = [float(bias_val >> total_shift) for bias_val in scaled_bias]
model.fc1.bias.data = torch.tensor(bias_shifted, dtype=torch.float32)
# 2. model.fc1.bias.data = torch.tensor([11*256*128*128/((2**t)*(2**k)), -7*256*128*128/((2**t)*(2**k)), -18*256*128*128/((2**t)*(2**k))], dtype=torch.float32)
# 3. verilog에 집어넣어져있는 값(bias 잘못된 값)
# model.fc1.bias.data = torch.tensor([11*256, -7*256, -18*256], dtype=torch.float32)




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





# # *********************** c. mem파일 연산 결과   ************************** #
# # *************************##******************************************** #
# root_folder = r'C:\github\Braille_generator_FPGA\mem_outputs_dongeun'
# index_to_letter = ['a', 'b', 'c']
# # for letter in ['a', 'b', 'c']:
# for letter in ['a', 'b', 'c']:
#     for i in range(1, 2):  # 1 to 4
#         alphabet = f'{letter}_{i}'
#         mem_path = os.path.join(root_folder, f'{alphabet}_gray.mem')
#
#         try:
#             # ✅ 새로운 input tensor 생성
#             # input_tensor = create_cyclic_input(28, 28, 1)
#             input_tensor_mem = load_mem_as_image(mem_path)  # shape: [1, 1, 28, 28]
#
#
#             # print("==============================================================")
#             # print("🔹 새로운 Input Feature Map 정보:")
#             # print(f"Shape: {input_tensor_mem.shape}")
#             # print(f"Min value: {input_tensor_mem.min().item()}")
#             # print(f"Max value: {input_tensor_mem.max().item()}")
#
#
#
#             # 일부 값들 확인 (좌상단 5x5 영역)
#             # print("\n Input Feature Map 좌상단 5x5 영역:")
#             # print(input_tensor_mem[0, 0, :29, :29])
#
#             print(f"\n📄 Input mem 연산: {alphabet}")
#             # # ✅ 이제 이 input_tensor로 연산 수행
#             # print("\n" + "="*80)
#             # print("🔹  Input으로 연산 시작")
#             # print("="*80)
#
#
#             #
#             # ✅ Conv1 연산
#             conv1_out = model.conv1(input_tensor_mem)
#             print("🔹 [Conv1 출력] shape:", conv1_out.shape)
#             for i in range(conv1_out.shape[1]):
#                 print(f"Conv1 채널 {i} 값:")
#                 print(conv1_out[0, i])
#
#
#
#             # ✅ ReLU 적용
#             relu1_out = F.relu(conv1_out)
#             print("\n🔹 [ReLU1 출력] (Conv1 → ReLU)")
#             for i in range(relu1_out.shape[1]):
#                 print(f"ReLU 채널 {i} 값:")
#                 print(relu1_out[0, i])
#
#             # float → int 변환 (예: 32비트 정수)
#             relu1_int = relu1_out.to(torch.int32)
#
#             # 8비트 오른쪽 shift
#             relu1_shifted = relu1_int >> t
#
#             # (선택) float로 다시 변환
#             relu1_shifted = relu1_shifted.to(torch.float32)
#
#
#
#
#
#             # ✅ MaxPooling1 적용
#             pool1_out = F.max_pool2d(relu1_shifted, 2, 2)
#             print("\n🔹 [MaxPool1 출력] shape:", pool1_out.shape)
#             for i in range(pool1_out.shape[1]):
#                 print(f"MaxPool1 채널 {i} 값:")
#                 print(pool1_out[0, i])
#
#
#
#
#             # ✅ Conv2 연산
#             conv2_out = model.conv2(pool1_out)
#             print("\n🔹 [Conv2 출력] shape:", conv2_out.shape)
#             for i in range(conv2_out.shape[1]):
#                 print(f"Conv2 채널 {i} 값:")
#                 print(conv2_out[0, i])
#
#
#
#             # ✅ ReLU2 적용
#             relu2_out = F.relu(conv2_out)
#             print("\n🔹 [ReLU2 출력] (Conv2 → ReLU)")
#             for i in range(relu2_out.shape[1]):
#                 print(f"ReLU2 채널 {i} 값:")
#                 print(relu2_out[0, i])
#
#             # float → int 변환 (예: 32비트 정수)
#             relu2_int = relu2_out.to(torch.int32)
#
#             # 8비트 오른쪽 shift
#             relu2_shifted = relu2_int >> k
#
#             # (선택) float로 다시 변환
#             relu2_shifted = relu2_shifted.to(torch.float32)
#
#
#             # ✅ MaxPooling2 적용
#             pool2_out = F.max_pool2d(relu2_shifted, 2, 2)
#             print("\n🔹 [MaxPool2 출력] shape:", pool2_out.shape)
#             for i in range(pool2_out.shape[1]):
#                 print(f"MaxPool2 채널 {i} 값:")
#                 print(pool2_out[0, i])
#
#             # ✅ Flatten (벡터화)
#             flatten_out = pool2_out.view(pool2_out.size(0), -1)
#             print("\n🔹 [Flatten 출력] shape:", flatten_out.shape)
#             print("Flatten 값:")
#             print(flatten_out[0])  # 배치 크기 1이므로 [0]만 출력
#
#
#             # ✅ FC1 연산 (Fully Connected Layer)
#             fc1_out = model.fc1(flatten_out)
#
#
#             # float → int 변환 (예: 32비트 정수)
#             fc1_int = fc1_out.to(torch.int32)
#
#             # 8비트 오른쪽 shift
#             fc1_shifted = fc1_int >> 0
#
#             # (선택) float로 다시 변환
#             fc1_shifted = fc1_shifted.to(torch.float32)
#
#             # ✅ 예측 클래스 인덱스 → 알파벳으로 변환
#             pred_index = torch.argmax(fc1_shifted[0]).item()
#             pred_letter = index_to_letter[pred_index]
#
#             print(f"\n📄 Input mem: {alphabet}")
#             print(f"🔹 FC1 Output Vector: {fc1_shifted[0].tolist()}")
#             print(f"✅ Predicted Letter: '{pred_letter.upper()}' (index {pred_index})")
#         except Exception as e:
#             print(f"Failed to process {mem_path}: {e}")
#
#
# ⬇ 모델 GPU로 보내기
model = model.to(device)

# ⬇ 정확도 출력
accuracy = evaluate(model, test_loader)
print(f"\nTest Accuracy: {accuracy:.2f}%")
#
# weight_bias_path = r'C:\github\Braille_generator_FPGA\test'
# pixel_scale = 256
# scale = 128  # 보통 -1.0 ~ 1.0 사이의 값이면 128 곱해서 int8 사용
#
#
# # conv1.bias
# bias = model.conv1.bias.data.clone().cpu()
# print("\nconv1_bias.shape:", bias.shape)
# # 정수 변환 → int16 clamp
# int_bias = bias.round().to(torch.int32).clamp(-32768, 32767).to(torch.int16)
# print("conv1_int_bias.shape:", int_bias.shape)
# with open(os.path.join(weight_bias_path, "conv1_bias.mem"), "w") as f:
#     for out_ch in range(int_bias.shape[0]):
#         val = int_bias[out_ch].item()
#         hex_val = f"{(val & 0xFFFF):04x}"  # 2-digit hex (8bit signed)
#         f.write(f"0x{hex_val}\n")
# print("conv1_bias_write_done")
#
#
# # conv2.bias
# bias = model.conv2.bias.data.clone().cpu()
# print("\nconv2_bias.shape:", bias.shape)
# int_bias = bias.round().to(torch.int32).clamp(-32768, 32767).to(torch.int16)
# print("conv2_int_bias.shape:", int_bias.shape)
# with open(os.path.join(weight_bias_path, "conv2_bias.mem"), "w") as f:
#     for out_ch in range(int_bias.shape[0]):
#         val = int_bias[out_ch].item()
#         hex_val = f"{(val & 0xFFFF):04x}"  # 2-digit hex (8bit signed)
#         f.write(f"0x{hex_val}\n")
# print("conv2_bias_write_done")
#
#
# # fc1.bias
# bias = model.fc1.bias.data.clone().cpu()
# print("\nfc1_bias.shape:", bias.shape)
# int_bias = bias.round().to(torch.int32).clamp(-32768, 32767).to(torch.int16)
# print("fc1_int_bias.shape:", int_bias.shape)
# with open(os.path.join(weight_bias_path, "stage3_fc1_bias.mem"), "w") as f:
#     for out_ch in range(int_bias.shape[0]):
#         val = int_bias[out_ch].item()
#         hex_val = f"{(val & 0xFFFF):04x}" # 2-digit hex (8bit signed)
#         f.write(f"0x{hex_val}\n")
# print("fc1_bias_write_done")
#
#
#

print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹여기서 부터는 이미지로 검증🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
