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


# class CNN(nn.Module):
#     def __init__(self, ch1, ch2):
#         super(CNN, self).__init__()
#
#         # pixel 데이터 추가
#         self.conv1 = nn.Conv2d(1, ch1, kernel_size=5, padding=0, bias=True)
#         # 합성곱 레이어
#
#         self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max pooling
#         # MaxPool2d(2, 2): 2x2 최대 풀링 → 크기를 절반으로 줄임
#
#         self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=5, padding=0, bias=True)
#         self.dropout = nn.Dropout(p=0.25)  # 🔸 Dropout 확률 25% 추천
#         self.fc1 = nn.Linear(ch2 * 4 * 4, 26, bias=True)  # 완전연결,  # a,b,c 분류
#         # self.fc2 = nn.Linear(32, 3)  # a,b,c 분류
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         ##soft max한거
#         x = self.fc1(x)
#
#         return x
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
    transforms.Lambda(lambda t: (t * 256).clamp(0, 255).round().to(torch.uint8)),
    transforms.Lambda(lambda t: t.to(torch.float32))  # ✅ 다시 float으로 바꿔줌
    ])



# ************** a-1. EMNIST 전체 훈련/테스트 데이터셋 ************** #
# *************************************************************** #

# EMNIST 데이터셋 불러오기 (예: 'letters')
train_dataset = EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

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

model = CNN(3,6)

# Conv1 weights
model.conv1.weight.data = torch.tensor(    [
    [
        [
            [23, 33, 5, -25, -41],
            [45, 19, 40, 18, -94],
            [55, 30, 54, -4, -100],
            [74, 36, 35, -14, -98],
            [59, -18, -27, -53, -34],
        ],
    ],
    [
        [
            [-9, -35, -75, -79, -33],
            [-47, -50, 5, -27, -81],
            [36, 17, 54, -8, -45],
            [34, 48, 74, 23, -18],
            [55, 23, 49, 25, 6],
        ],
    ],
    [
        [
            [22, 30, 21, 12, -25],
            [33, 18, 39, 15, 16],
            [42, 28, 36, 6, 28],
            [44, 39, 47, 17, 1],
            [37, 43, 18, 4, -20],
        ],
    ],
]

, dtype=torch.float32)
# Conv2 weights
model.conv2.weight.data = torch.tensor(  [
    [
        [
            [-7, -17, -9, 27, -13],
            [27, -29, 43, 32, -41],
            [-13, -5, 62, 67, -32],
            [-40, 7, 56, 59, 11],
            [-21, -9, -13, 18, 32],
        ],
        [
            [33, -18, -11, 7, -13],
            [-1, -2, 6, -19, -45],
            [8, -35, 6, 0, -61],
            [76, -54, -27, 12, -26],
            [10, -27, 6, 23, -6],
        ],
        [
            [-9, -2, -20, -10, -12],
            [3, -14, -4, -38, 19],
            [-8, 10, -28, -24, 23],
            [-12, 1, 0, 2, 33],
            [-5, -5, 2, 34, 26],
        ],
    ],
    [
        [
            [-4, -39, -52, -42, -26],
            [-5, -43, -22, -6, 0],
            [-3, 16, 11, 8, 9],
            [-8, 24, 7, -8, 6],
            [-23, -3, -53, -64, -38],
        ],
        [
            [-25, -1, 22, 16, 20],
            [-55, -18, 24, 45, 5],
            [-112, -72, -10, 39, -10],
            [-38, -9, 33, 30, 16],
            [39, 51, 56, 42, -4],
        ],
        [
            [20, -2, 2, 17, 11],
            [36, 17, 7, 12, 16],
            [45, 2, -7, 6, 3],
            [-5, -22, -23, -25, -25],
            [-12, -24, -25, -5, 6],
        ],
    ],
    [
        [
            [0, -14, 2, 32, 8],
            [17, 16, -7, 19, 10],
            [19, -13, -51, 10, 27],
            [21, 0, -19, 37, 43],
            [39, 26, 13, 4, 14],
        ],
        [
            [-46, -53, -33, -47, -75],
            [-13, -17, -76, -39, -37],
            [68, -6, -96, -107, -72],
            [71, 34, -68, -127, -46],
            [28, 13, -10, -37, -10],
        ],
        [
            [9, 11, 16, 3, 12],
            [-3, 23, 24, -6, 9],
            [-22, -22, 3, -4, 13],
            [-17, -49, -20, -22, 28],
            [-28, -32, -23, -22, 31],
        ],
    ],
    [
        [
            [32, 22, 34, -16, -35],
            [45, 21, -31, -45, 30],
            [-2, -25, -33, 7, -1],
            [-13, -3, -7, 15, 1],
            [-3, 2, 21, 19, -21],
        ],
        [
            [5, -88, -124, 1, 121],
            [-85, -19, 78, 83, 29],
            [-18, 6, 17, -18, -25],
            [22, -16, -2, 22, 20],
            [29, 7, 31, 32, 13],
        ],
        [
            [-26, 3, 0, 2, -41],
            [-1, 10, 16, -38, -22],
            [12, 4, -2, -11, -4],
            [20, 18, 7, 3, 16],
            [11, -21, -24, -25, -10],
        ],
    ],
    [
        [
            [-22, -39, -33, -3, 18],
            [-14, -35, -49, -29, 8],
            [-16, 5, -34, -41, 4],
            [-39, 4, -3, -13, -15],
            [13, 18, 12, 3, 4],
        ],
        [
            [-16, -29, -15, -17, 0],
            [28, 27, 15, -1, -19],
            [0, 31, 29, 29, 6],
            [-11, 56, 84, 55, -10],
            [-73, -40, -10, 29, 19],
        ],
        [
            [-7, 3, 13, 1, -18],
            [1, -4, -5, 3, -2],
            [15, -13, -12, -21, -11],
            [25, 15, -22, -6, -31],
            [36, 28, 8, 5, 2],
        ],
    ],
    [
        [
            [0, 4, 10, 11, -55],
            [20, 33, 25, 10, -61],
            [40, 45, 30, 20, -14],
            [15, 27, 30, -29, -25],
            [4, -11, 0, -38, -6],
        ],
        [
            [-32, 19, 2, -24, -23],
            [-5, -22, -37, -28, 57],
            [-43, -63, -59, -35, 20],
            [-31, -40, -40, -27, 4],
            [43, -2, -32, -26, 30],
        ],
        [
            [28, -7, -26, 0, 4],
            [1, -37, -38, -34, -2],
            [4, -22, -16, -20, 3],
            [-10, -4, 12, 3, 18],
            [14, 23, 37, 18, 30],
        ],
    ],
]

, dtype=torch.float32)

model.fc1.weight.data = torch.tensor(   [
    [-16, -6, 0, -7, -4, 1, 4, -1, 3, 0, 4, 2, -11, 0, 8, 10, 4, 1, -3, -12, 9, 0, 1, 0, 7, 0, -2, 3, -4, -6, -6, -2, -18, 0, 4, 6, -15, 0, 5, 1, 10, -8, -1, 4, 6, -12, -8, 0, -1, 7, 8, 3, 5, 3, 6, 0, 7, 0, 0, -13, 3, 4, 7, 4, -1, 2, 1, 0, -16, -5, 0, -6, -5, -3, 7, 3, 6, 0, 1, 7, -1, -3, -11, -18, 0, -5, -1, 0, 3, 1, 4, 5, 8, 0, 0, 3],
    [3, 3, -5, -6, 5, 2, -4, -13, 4, -3, 2, 12, 6, -8, 6, 16, -15, 1, 2, -5, -16, 0, 3, 0, -21, -6, 10, 2, 1, 7, 9, 6, 5, 0, -9, -5, -2, 5, -2, 3, -13, -4, 2, 11, -10, 0, -14, 0, 6, 6, 3, -8, -6, -4, -2, 7, 1, 1, 4, 18, 3, 9, 7, -5, -16, -8, -2, -7, -15, -5, 6, 9, -15, -3, 4, 9, -2, 0, -6, -8, -2, -2, 0, -1, -1, 0, 3, -6, -2, 0, -12, -14, 0, 0, 9, 7],
    [-12, -8, -4, -13, 3, 4, 4, -21, 2, -7, -9, -38, -6, -42, -31, -30, 5, 4, 2, -2, 0, -18, -25, 6, -3, 4, 3, 3, 0, 10, 9, 10, -34, -5, 8, 5, -14, -8, 4, -8, 7, 4, -10, -11, 13, 10, 4, -6, 0, 6, 9, 2, 4, 8, 12, -5, 4, -1, -22, -5, -5, -7, -10, -9, -31, -1, 0, -2, -37, -5, -1, -7, 9, -5, -15, -23, 11, 6, 0, -3, 7, 6, 5, 11, 5, 4, 2, 5, 0, -11, -31, -22, -20, -35, -34, -20],
    [-9, -3, 3, 2, 0, -3, 4, 6, 4, 0, 1, 10, 1, 1, 0, 5, -9, -3, -4, -11, 10, 5, 1, -20, 11, 7, 1, 0, 5, 6, 6, 3, -30, -12, -1, 2, -18, -3, 8, 9, -4, 6, 5, 7, 4, 4, 6, 8, -8, -13, -21, -13, -15, -12, -14, -16, 2, 0, -10, -15, 9, 7, -1, 13, 0, 0, 4, 11, -10, 1, 8, 15, -20, -6, -3, -7, -2, -9, -7, -8, -7, -4, -4, 7, -5, 0, 4, 5, 1, -2, 1, 8, 3, -1, 1, 7],
    [-8, -9, -3, -10, 0, 6, 5, -3, -2, -30, -25, -31, -8, -27, -20, -28, 4, 3, 0, -4, 5, 7, 9, 1, -6, 0, 0, 2, 1, 11, 10, 9, -6, -3, 6, 3, 1, -5, -10, -17, -22, -17, -24, -20, -5, 5, -7, -10, 9, 15, 16, 0, 7, 8, 6, -13, 4, 3, 0, -11, 5, 8, -1, 0, -14, -10, -3, -2, 2, 10, 3, -8, 12, 3, 0, -14, 13, 6, 0, -3, 3, 3, -5, -7, 3, 0, -1, -7, -3, -4, -20, -23, -13, -24, -37, -21],
    [11, 8, 2, -7, 5, 8, -1, -25, 0, -5, -22, -41, 4, 3, -16, -56, -6, 0, 0, -1, 10, 11, 13, 8, 16, 9, 8, -1, -15, -15, -24, -35, -19, -29, -43, -11, -18, -4, -5, -12, -14, -3, -2, -3, -24, -11, -8, -5, 7, 16, 20, 18, -4, 2, 5, -4, -6, -1, -7, -44, 2, 1, -11, -44, -8, -6, 7, 9, 9, 3, 8, 8, 14, 5, 6, 5, -1, -3, -4, -6, 6, 11, 10, -13, -3, -1, -7, -42, -1, -1, -13, -45, 1, 2, -3, -32],
    [7, -1, 0, -14, 1, -5, 2, -13, -5, -4, 1, 0, -4, 1, 3, 11, -2, 1, -1, -9, -8, -1, 3, 13, 1, 3, 7, 8, 6, 5, 1, -7, -38, -19, -3, 0, -2, -13, -10, 0, 0, -11, -3, 2, 6, 2, -11, -1, 11, 16, 16, 6, 8, 8, 4, -8, 0, -1, 0, -1, 0, 6, 7, -5, -6, 0, -3, -5, 8, 4, 0, -3, 7, 5, 3, 2, 4, -3, -7, -11, 5, 3, 3, 0, -3, -11, -7, -11, -16, -15, -2, -3, -20, 4, 10, 6],
    [0, 4, 2, 1, 0, 0, -1, -1, 1, -3, 6, 10, 3, 5, 11, 18, -10, -4, -5, -6, 0, 5, 4, 4, -5, -3, 11, 5, -4, -42, -34, 0, 12, 4, -2, -6, 5, 2, 0, 5, -1, 0, 2, 7, -10, -19, 9, 11, -41, -51, -35, -27, -33, -14, -15, -9, 0, 1, 1, -9, 1, 5, 6, -14, -18, -39, -26, -48, -14, -3, 10, 10, -16, 1, 1, 0, -26, -10, 0, -1, 6, 0, 1, 4, 9, 1, -1, -2, 6, 2, -4, 0, 9, 8, 4, 8],
    [-1, 0, -15, -47, -3, 0, -21, -23, -2, 3, -13, -17, -3, 6, -23, -16, 0, 2, 3, 7, -26, 0, -3, -20, 4, -1, 0, 5, 18, 5, 6, 5, -25, 0, 6, 3, -23, -7, 11, 5, -15, 3, 7, 4, -18, 4, 8, 3, -11, -7, -15, -13, -3, -8, -5, -10, -39, -5, -2, -23, -22, -10, -16, -2, 1, -2, 0, 7, -9, -15, -19, -20, -34, -2, 0, -23, 0, 5, -3, 4, -40, 4, 5, 5, -35, 10, 8, 11, -5, 10, 10, -6, 1, 7, 10, -16],
    [-13, 0, -1, -16, 0, 8, 3, -11, 6, 9, 2, 0, 0, 3, 0, 5, 4, 2, 0, 5, 6, -3, -8, -1, 17, 6, -6, -26, 17, 13, 6, -6, -43, -19, -10, -6, -13, -16, 7, 8, 15, 4, 2, 10, 21, 9, -2, -1, -6, -6, -12, 5, 3, -10, -19, -2, -10, -27, -32, -14, -3, -8, -8, -17, 10, 3, 2, 4, 7, -1, -8, -7, 6, -15, -9, -1, 13, 3, -5, -34, -77, -34, -18, -9, -56, -8, -4, 1, -21, 0, 2, 7, -13, 4, 11, 15],
    [9, 19, 9, -6, 4, 0, 0, -30, -1, 0, 1, -21, 3, 9, 5, 7, -15, 1, 7, 6, -13, -4, 4, 0, -21, -2, -2, -9, -16, -5, 4, 9, 14, 9, -2, -24, 3, 0, -14, -13, -2, -1, -8, 1, -16, 0, 12, 15, -45, -37, -18, -3, -4, 2, 5, 9, 3, 5, 10, 10, 0, 1, 5, 0, -33, -31, -12, -20, -15, 0, 3, -9, -5, 5, 6, 0, -12, 7, 8, 12, 9, 3, 3, 6, 5, -1, 0, -5, 2, -6, -4, -24, 5, 0, -19, -37],
    [1, 4, -4, -33, -3, 4, -11, -44, -4, 6, -36, -39, 0, 7, -30, -22, -19, -13, -6, -2, -17, -6, -13, -17, 6, 6, 1, 11, 12, 9, 13, 12, -1, 1, 6, 3, -6, -6, 6, -5, -1, 1, 2, -2, -4, 11, 6, 7, -42, -32, -39, -27, -40, -22, 0, 7, -16, -6, 8, 2, -10, -12, -20, 2, -13, -13, -50, -73, -28, -17, -12, -46, -40, -12, -8, -12, -4, 3, 1, 4, 1, 6, 6, 7, 5, 9, 6, 10, 7, 6, 7, -16, 6, 8, 7, -6],
    [-3, -3, 1, 5, 7, 10, 4, 21, 11, 14, 7, 28, 1, 0, 8, 23, 9, 0, -1, 2, -1, -6, -4, -2, -21, -23, -19, -33, -74, -44, -53, -56, 9, 9, 9, 0, 6, 9, 2, 1, -8, 0, -21, -11, -16, -42, -32, -23, -16, -15, -21, -6, 5, -1, -5, 12, 8, -6, 7, 25, 1, -17, 12, 13, -9, 3, -10, -7, -9, 1, -3, -1, -13, 0, -3, -2, -26, -2, -10, -4, 4, -8, -15, -6, 5, -7, 0, -7, 3, -10, 3, 0, -10, -21, -17, -1],
    [-10, -8, 1, 12, -7, 0, 7, 17, -3, 3, 6, 16, -1, 0, 0, 11, 7, 1, -4, 0, 7, -7, -15, -9, 0, -25, -15, -4, -44, -49, -21, -10, 10, 7, 6, 6, 8, 10, 10, 3, 5, 8, 4, -1, -10, -11, -2, -2, -11, -13, -13, -8, -1, -5, -3, 3, 4, -1, 0, 0, 6, -13, -11, 12, -1, 6, 1, 1, -6, -5, -2, 7, -20, -6, -4, -2, -18, -2, 1, 8, 4, -3, -21, -5, 9, 4, -13, 4, 7, 5, -1, 15, 5, -4, -7, 6],
    [-25, -21, 0, 0, -9, -13, 4, 9, 1, -14, 3, 11, 3, -22, -3, -2, 2, 1, 0, 0, 0, -6, -24, -19, -5, 1, 3, -2, 0, 7, 3, -1, -27, -9, 2, 7, -19, -16, -6, 9, 4, 3, -31, -15, 9, 6, -14, -81, -4, 4, 4, 1, 4, 3, 6, 3, 6, 5, -6, -25, 8, 1, 1, 2, -31, -6, 1, 6, -26, -4, 1, 5, 3, -9, -19, -13, 9, 1, -2, -24, 4, 5, 2, -20, 0, 2, -11, -2, 0, -8, 5, 7, -12, -14, 6, 7],
    [4, 0, 14, 21, 3, 3, 7, 12, 4, -6, -13, -14, 5, -11, -42, -54, -8, 0, 7, -1, -7, 3, 11, 5, -16, 0, 5, 0, -45, -30, -25, -44, -7, -6, -10, 8, -5, 0, -13, -9, -8, -9, -10, -49, -12, 0, 5, -25, 14, 15, 15, 11, -3, 1, 2, 6, -6, 3, 9, 17, -5, -4, 0, 7, -5, 2, 7, 6, -5, 0, 12, -5, -9, 1, 8, -1, -22, -7, -2, -17, 0, 2, -2, 3, -1, 0, -1, 9, 1, 3, -17, 3, 4, 5, -22, -46],
    [3, -7, 3, 2, -1, 0, 0, 9, -3, 4, 2, 8, 0, 3, 4, 5, -2, -2, -4, -15, -6, -2, 0, 0, -6, 0, 0, -4, -6, 1, 0, -8, -32, -10, -3, 0, -9, -7, -1, 0, 5, -6, 4, 6, 1, -14, 0, 4, 12, 14, 16, 5, 10, 7, 0, -13, 3, -3, -3, -13, -6, 0, 3, 12, -10, 0, 0, 1, 10, 5, -1, 0, 10, 4, 0, -4, 6, -3, -4, 4, 6, 1, 7, -2, -2, -9, -1, -7, -15, -4, -3, 1, -15, 5, 2, 0],
    [-2, -6, 0, 6, -2, -9, -1, 7, -5, -1, -8, -21, 1, 8, -1, -35, 0, 3, 4, 4, -7, 1, 3, -1, -17, -16, -15, -11, -40, -30, -9, 0, 10, 8, -1, 5, 4, 4, -4, -4, 5, 9, -1, -6, 0, 6, 0, 6, 7, 7, 5, 5, -2, 2, 4, -1, 1, 2, 8, 7, 0, -2, 0, -13, -1, 0, 8, 10, -4, -4, 3, 11, -10, -1, 1, -1, -11, 2, 4, 11, 6, 3, 0, -14, 7, 4, 3, -4, 5, 5, -11, -29, 6, 3, -21, -48],
    [6, -5, -22, -21, 0, -6, -17, -29, -13, -9, -3, 0, -11, -12, -4, 14, -7, 1, 0, 0, -9, 0, -1, 0, 15, 4, 3, -5, 16, 13, 8, -1, -28, -23, -19, -8, -28, -6, -7, 3, -1, 1, 9, 13, 3, -3, -10, -4, 3, 10, 11, 10, 5, 13, 11, 13, -10, -9, -22, -48, 1, 0, -6, -19, -5, -7, 0, 5, 20, 7, 7, 7, 17, 11, 6, 7, 11, 4, -6, -11, 0, 10, 9, -15, -26, -3, -1, -51, -56, -24, -3, -8, -33, 0, 12, 11],
    [-1, 11, 0, -43, 1, 5, -3, -58, 1, 0, -13, -53, 1, 8, -4, -35, 12, 2, 0, 4, 15, 5, 5, 7, 1, -3, 5, 0, -13, -4, 4, 3, -10, -4, -1, -7, -11, -8, -3, -2, -18, -8, -5, -1, -10, 5, -1, 5, 1, -7, -6, 5, 0, -12, -9, -4, -2, -11, -6, -24, -3, -8, -9, -39, 19, 10, 6, 15, 17, 3, 4, 10, 16, 3, 0, 4, -13, -3, -13, -12, -32, 9, 4, 1, -7, 7, 4, 0, 3, 3, 2, 0, 7, 4, 4, -2],
    [8, 7, 6, 2, 4, 3, 1, 4, 0, 1, 0, 6, -13, -3, 0, 0, 6, -4, -10, 5, -6, -12, -12, -11, 6, 0, -3, 1, 2, 2, -2, 0, 15, 7, 6, 2, 14, 0, 3, 5, 15, 0, -1, 5, 14, 4, -2, 0, -29, -54, -31, -10, -12, -14, -21, -1, 2, -5, 4, 9, -4, 0, 6, 11, 1, -26, -23, -12, -25, -50, -36, -11, -1, -16, -1, 3, 7, 0, 3, 9, 9, 1, -5, 4, 10, 1, 3, 11, 5, -12, 6, 12, -20, -20, 0, 10],
    [6, 0, 6, 8, 4, 5, 4, 4, 5, 8, 1, 2, -4, -9, -15, -9, 8, -7, -1, 9, -14, -8, -6, 0, -6, -2, -11, -17, -13, -14, -20, -26, 19, 13, -2, 0, 14, 9, -3, -4, 19, 7, 2, -4, 15, 5, 3, -7, -7, -52, -33, -2, -7, -18, -4, 8, -9, -11, 3, 19, -12, -7, 0, -5, 13, -2, -24, -28, 3, -7, -26, -18, -5, -6, -7, -34, -1, 0, 0, -13, 3, -17, 1, 9, 4, -10, 4, 9, 1, 0, 4, 13, 2, -1, -3, 13],
    [2, 1, -5, 16, 8, 13, 8, 23, 8, 9, 10, 21, 4, 3, -2, 1, -4, -2, -10, -4, -2, 1, -2, -4, 8, 0, -6, -2, 1, -14, -17, -6, 21, 1, 7, -6, 19, 1, 12, -4, 15, 1, 6, -7, 7, -10, 1, -9, -51, -48, -51, -28, -13, -12, -17, -6, 0, -3, -1, 10, 1, 2, -8, 9, -19, -42, -10, -5, -23, -31, -5, 1, -11, -12, 1, -9, 0, -3, 3, 3, -2, -21, -24, -9, -3, -6, -9, 1, -3, 0, -4, 12, -1, -3, -10, 6],
    [10, 12, 5, 3, 1, -3, -5, -21, -12, -8, 1, -11, -8, 2, 4, 0, -4, -5, 0, 4, -8, -17, -13, -13, 6, -13, -9, -3, 14, -9, -7, 1, 15, 10, 9, -14, 9, 9, 1, -4, -22, 10, -1, 2, -31, -5, 13, 9, -12, -18, -18, 7, -6, -11, -2, 10, -2, 2, 4, 7, 7, 7, 8, 1, 10, 15, 10, -25, 8, 4, 2, -15, -3, 0, 6, 9, -27, 1, 7, 9, -37, -21, 4, 9, -15, -5, 3, 3, 1, 1, -1, -20, 8, 5, -1, -38],
    [19, 10, 9, 8, 3, -2, 0, -12, -3, -2, -4, -5, -5, -2, -3, -5, -12, -4, 2, -1, -19, -2, -2, -19, 7, 1, -6, -31, 10, 0, -1, -18, 18, 12, 8, -1, 7, 6, 0, 1, -7, -7, -10, -2, -8, -7, -13, -12, -26, -15, -20, 11, -7, -3, 0, 8, -10, -10, 0, 11, -12, -12, -12, -14, 12, 18, 9, -36, 10, 11, 6, -20, 14, 10, 0, -20, -11, -6, -22, -31, 13, -17, 4, 13, -12, -2, 2, 4, -10, 3, 2, 5, 6, 5, 0, 8],
    [-19, -9, 1, -15, -18, -5, -3, -27, -11, -16, -26, -36, -3, -19, -16, -27, 7, 0, -2, -12, 7, -2, -3, 2, 7, 4, 4, 9, 5, 9, 11, 14, -5, 6, 5, 2, -27, -6, -7, -15, -28, -10, -20, -14, -2, 7, -2, 2, 13, 13, 4, -47, -5, -8, -12, -11, -1, 4, 8, 6, 7, 8, 10, 7, 15, 12, 9, -8, 13, -1, -10, -9, 0, -8, -4, 2, -6, 0, 2, 7, -48, -20, -8, -8, -27, 5, 6, 9, 3, 5, 3, -7, 1, 0, 2, -17],
], dtype=torch.float32)


conv1_bias_ints= torch.tensor(   [14, 52, -96, ]
, dtype=torch.float32)
scale = 256
scaled_bias = [x * scale for x in conv1_bias_ints]
model.conv1.bias.data = torch.tensor(scaled_bias, dtype=torch.float32)




# # conv2 bias , need (8 + (log2(scale) + 1)-t )bit,  16-t bit
# 1. Bitshift t만큼 한 결과를 float으로 저장 (PyTorch는 float만 허용)
conv2_bias_ints =torch.tensor(   [-35, -7, 2, -14, -23, 1 ]
, dtype=torch.float32)
# 각 요소에 대해 256 * 128 * 128을 곱하기
scale = 256 * 128
scaled_bias = [x * scale for x in conv2_bias_ints]
bias_shifted = [float(int(bias_val) >> t) for bias_val in scaled_bias]
model.conv2.bias.data = torch.tensor(bias_shifted, dtype=torch.float32)


# # total_shift need at leas 7bit
# # fc1 bias , need (8 + 2*(log2(scale) + 1) - total_shift )bit, 23 - total_shift bit
fc1_bias =  torch.tensor(  [-41, -35, 28, -23, -16, 21, 2, 0, 31, 52, -15, 53, 9, -11, -28, 0, -14, -2, -18, 17, -22, 8, -7, -52, 26, 0, ]
, dtype=torch.float32)
# 각 요소에 대해 256 * 128 * 128을 곱하기
scale = 256*128*128
scaled_bias = [x * scale for x in fc1_bias]
bias_shifted = [float(int(bias_val) >> total_shift) for bias_val in scaled_bias]
model.fc1.bias.data = torch.tensor(bias_shifted, dtype=torch.float32)


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

#
# model = CNN(3, 6)
# model.load_state_dict(torch.load("cnn_c3_6_ep10.pth"))
# model.eval()

# ⬇ 모델 GPU로 보내기
# model = model.to(device)

# ⬇ 정확도 출력
accuracy = evaluate(model, test_loader)
print(f"\nTest Accuracy: {accuracy:.2f}%")




weight_bias_path = r'C:\github\Braille_generator_FPGA\weight_bias_a_z_mem'
# conv1.bias
bias = model.conv1.bias.data.clone().cpu()
print("\nconv1_bias.shape:", bias.shape)
# 정수 변환 → int16 clamp
int_bias = bias.round().to(torch.int32).clamp(-128, 127).to(torch.int16)
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
int_bias = bias.round().to(torch.int32).clamp(-128, 127).to(torch.int16)
print("conv2_int_bias.shape:", int_bias.shape)
with open(os.path.join(weight_bias_path, "conv2_bias.mem"), "w") as f:
    for out_ch in range(int_bias.shape[0]):
        val = int_bias[out_ch].item()
        hex_val = f"{(val & 0xFFFF):04x}"  # 2-digit hex (8bit signed)
        f.write(f"0x{hex_val}\n")
print("conv2_bias_write_done")


# fc1.bias
bias = model.fc1.bias.data.clone().cpu()
print("\nfc1_bias.shape:", bias.shape)
int_bias = bias.round().to(torch.int32).clamp(-128, 127).to(torch.int16)
print("fc1_int_bias.shape:", int_bias.shape)
with open(os.path.join(weight_bias_path, "stage3_fc1_bias.mem"), "w") as f:
    for out_ch in range(int_bias.shape[0]):
        val = int_bias[out_ch].item()
        hex_val = f"{(val & 0xFFFF):04x}" # 2-digit hex (8bit signed)
        f.write(f"0x{hex_val}\n")
print("fc1_bias_write_done")
#
#
#
print("fc1 output size:", model.fc1.out_features)
print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹여기서 부터는 이미지로 검증🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
print("\n🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def print_cnn_intermediate_outputs(model, input_tensor):
    x = input_tensor.clone()

    print("Input:", x.shape)

    print("conv1 weights:")
    print(model.conv1.weight.data)

    x = model.conv1(x)
    print("After conv1:", x[0, 0, :5, :5])

    x = F.relu(x)
    print("After ReLU1:", x[0, 0, :5, :5])

    x = model.pool(x)
    print("After pool1:", x[0, 0, :5, :5])

    x = model.conv2(x)
    print("After conv2:", x[0, 0, :5, :5])

    x = F.relu(x)
    print("After ReLU2:", x[0, 0, :5, :5])

    x = model.pool(x)
    print("After pool2:", x)

    x = x.view(x.size(0), -1)
    print("After flatten:", x)

    x = model.fc1(x)
    print("After fc1:", x)

    return x



transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 255).round().to(torch.uint8)),
    transforms.Lambda(lambda t: t.to(torch.float32))  # ✅ 다시 float으로 바꿔줌
])

#
# model = CNN(3, 6)
# model.load_state_dict(torch.load("cnn_c3_6_ep10.pth"))
# model.eval()



root_folder = r'C:\github\Braille_generator_FPGA\handwritebold'
for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
    for i in range(1, 2):  # 1 to 4
        alphabet = f'{letter}_{i}'
        image_path = os.path.join(root_folder, f'{alphabet}.png')

        try:
            img = Image.open(image_path).convert("L")
            img_tensor = transform(img).unsqueeze(0)

            # CNN 결과 출력
            print(f"\n====== {alphabet}.png ======")
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_letter = chr(predicted.item() + ord('a'))

            print("Output logits:", output)
            print(f"Predicted: {predicted_letter}")

            # 중간 결과 출력
            # print_cnn_intermediate_outputs(model, img_tensor)

            # 이미지 시각화
            img_np = img_tensor.squeeze().numpy()
            plt.imshow(img_np, cmap="gray")
            plt.title(f"{alphabet}.png")
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
print()
