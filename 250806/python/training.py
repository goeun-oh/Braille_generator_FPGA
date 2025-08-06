# # *****************************설명********************************* #
# # a ~ e는 모델을 만들고 학습시키고 모델을 저장하는 과정
# # a-1은 EMNIST 전부, a-2는 (a,b,c 만) 훈련하는 코드. 모델 학습 시킬 때 둘 중 하나는 반드시 주석!
# # d.는 모델을 학습시킬 때 하이퍼 파라미터 튜닝을 설정해 줄 수 있음
# # f 부터는 저장된 모델의 weight, bias만 확인하는 과정
# # 만약 학습을 다 시킨적이 있고 weight, bias만 확인하려면 a~e 주석처리하고 하기
# # ***************************************************************** #

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# datasets : MNIST 같은 데이터 셋을 쉽게 가져오기 위함
# transforms : 이미지 전처리를 위한도구



from torch.utils.data import DataLoader
# DataLoader : 데이터를 batch 단위로 나눠서 처리(메모리 절약 + 빠름)

from PIL import ImageOps
from torchvision.datasets import EMNIST
import torch
torch.set_printoptions(threshold=float('inf'), sci_mode=False, precision=6)


#transform 옵션
transform = transforms.Compose([
    transforms.Lambda(lambda img: ImageOps.invert(img)),  # 색 반전
    transforms.Lambda(lambda img: ImageOps.mirror(img)),  # 좌우 대칭 복원
    transforms.Lambda(lambda img: img.rotate(90, expand=True)),  # 시계방향 90도 회전
    transforms.ToTensor(),
#transform 옵션
# MNIST는 기본적으 PIL 이미지(0~255) (Python Image Library)
# 이미지 열기 (.jpg, .png, .bmp, .gif, .tiff, ...)
# 크기 조절 (resize)
# 잘라내기 (crop)
# 회전, 흑백 변환, 필터 적용
# 픽셀 값 접근
# TOTensor()를 쓰면 ->[0.0, 1.0]범위의 텐서로 변환
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
# #
# #





# # # **************** a-2. A,B,C만  훈련/테스트 데이터셋 *************** #
# # # *************************************************************** #
# target_labels = list(range(1, 14))  # 1~13
#
# # 데이터셋 필터링 함수
# def filter_dataset(dataset):
#     indices = [i for i, (_, label) in enumerate(dataset) if label in target_labels]
#     filtered = Subset(dataset, indices)
#     return filtered
#
# # 전체 EMNIST 데이터셋 (transform은 기존 정의된 것 사용)
# full_train_dataset = EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
# full_test_dataset = EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
#
# # a~m만 필터링
# train_dataset = filter_dataset(full_train_dataset)
# test_dataset = filter_dataset(full_test_dataset)
#
# # Subset은 index만 들고 있으므로 custom Dataset으로 감쌈
# class AMDataset(torch.utils.data.Dataset):
#     def __init__(self, subset):
#         self.subset = subset
#
#     def __len__(self):
#         return len(self.subset)
#
#     def __getitem__(self, idx):
#         img, label = self.subset[idx]
#         # label: 1~13 → 0~12로 바꿈 (모델 출력과 index 맞추기 위해)
#         return img, torch.tensor(label - 1, dtype=torch.long)
#
# # dataset wrapping
# train_dataset = AMDataset(train_dataset)
# test_dataset = AMDataset(test_dataset)
#
# # dataloader
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# # # *************************************************************** #
# # # *************************************************************** #








# ********************* b. CNN 모델 만들기 ************************ #
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

        self.dropout = nn.Dropout(p=0.25)  # 🔸 Dropout 확률 25% 추천
        self.fc1 = nn.Linear(ch2 * 4 * 4, 26, bias=True)  # 완전연결,  # a,b,c 분류
        # self.fc2 = nn.Linear(32, 3)  # a,b,c 분류

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        ##soft max한거
        x = self.fc1(x)

        return x
# forward: 입력이 모델을 통과할 때의 연산 정의
# ReLU: 비선형 활성화 함수 → 딥러닝에서 매우 중요
# view: 텐서를 펼쳐서 FC 레이어에 넣음

# class CNN(nn.Module):
#     def __init__(self, ch1, ch2, ch3):
#         super(CNN, self).__init__()
#
#         # pixel 데이터 추가
#         self.conv1 = nn.Conv2d(1, ch1, kernel_size=5, padding=0, bias=True)
#         # 합성곱 레이어
#         self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max pooling
#         # MaxPool2d(2, 2): 2x2 최대 풀링 → 크기를 절반으로 줄임
#         self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=5, padding=0, bias=True)
#         self.conv3 = nn.Conv2d(ch2, ch3, kernel_size=3, padding=1, bias=True)
#
#         self.dropout = nn.Dropout(p=0.25)  # 🔸 Dropout 확률 25% 추천
#         self.fc1 = nn.Linear(ch3 * 2 * 2, 26, bias=True)  # 완전연결,  # a,b,c 분류
#         # self.fc2 = nn.Linear(32, 3)  # a,b,c 분류
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         ##soft max한거
#         x = self.fc1(x)
#
#         return x
# *************************************************************** #
# *************************************************************** #





# ******************* c.손실함수와 옵티마이저 정의 ****************** #
# ************************************************************** #

import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device : 모델과 데이터를 CPU, GPU 어디에 올릴지 정함

# ************************************************************** #
# ************************************************************** #







# # ********************** d.모델 훈련 루프 ************************ #
# # channel_configs => 첫 번째 kernerl 채널 수, 두 번째 kernel 채널 수
# # epoch_list => epoch 총 몇번 할건지
# # ex) epoch_list = [5,6,7] => channel_configs 각각에 대하여 epoch 5,6,7번 학습시키겠다
# # ************************************************************** #
# # 훈련 루프
# channel_configs = [
#     (3, 6),
#     (4, 4),
#     (4, 8),
#     (5, 5),
#     (5, 10)
#
# ]
# epochs_list = [5]
#
# for ch1, ch2 in channel_configs:
#     for ep in epochs_list:
#         print(f"\n▶️ Config: conv1={ch1}, conv2={ch2}, epochs={ep}")
#         model = CNN(ch1, ch2).to(device)
#
#         criterion = nn.CrossEntropyLoss()
#         # 예측 정갑 간의 차이 계산하는 함수
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#         # 옵티마이저는 모델의 weight를 업데이트
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#         for epoch in range(ep):
#             model.train()
#             total_loss = 0
#             for images, labels in train_loader:
#                 images, labels = images.to(device), (labels-1).to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             print(f"  Epoch [{epoch+1}/{ep}] Loss: {total_loss:.4f}")
# # *************************정확도 평가**************************** #
# # ************************************************************** #
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), (labels-1).to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)
#         acc = 100 * correct / total
#         print(f"✅ Accuracy: {acc:.2f}%")
#
#         # 🔽 모델 저장
#         filename = f"cnn_c{ch1}_{ch2}_ep{ep}_a_z.pth"
#         torch.save(model.state_dict(), filename)
#         print(f"📦 Model saved as: {filename}")
# # ************************************************************** #
# # ************************************************************** #




#
# # **************** f. 훈련 모델 weight, bias 확인 **************** #
# # ************************************************************** #
#
# # 나중에 훈련모델 weight만 확인하고 싶을 때

#원하는 모델 인덱싱으로 찾기
# ex) model.load_state_dict(torch.load("cnn_cX_X_epX.pth"))
# x로 되어있는곳에 원하는 숫자 넣기 ( ex) ch1 = 3, ch2 = 3, ep =5 이면 아래처럼)
model = CNN(3, 6)
model.load_state_dict(torch.load("cnn_c3_6_ep5_a_z.pth"))
model.eval()

# 모델 conv1의 weight
print("=== Conv1 Weights ===")
print(model.conv1.weight.shape)  # => torch.Size([16, 1, 3, 3])
print(model.conv1.weight)  # => 실제 값 출력



print("=== Conv2 Weights ===")
print(model.conv2.weight.shape)  # => torch.Size([16, 1, 3, 3])
print(model.conv2.weight)  # => 실제 값 출력

print("\n=== FC1 Weights ===")
print(model.fc1.weight.shape)  # ex) torch.Size([128, 1568])
print(model.fc1.weight)

print("\n=== bias ===")
print("conv1 bias:", model.conv1.bias.data)
print("conv2 bias:", model.conv2.bias.data)
print("fc1 bias:", model.fc1.bias.data)
# ************************************************************** #
# ************************************************************** #
