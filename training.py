
#데이터 불러오기
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
#datasets : MNIST 같은 데이터 셋을 쉽게 가져오기 위함
#transforms : 이미지 전처리를 위한도구

from PIL import Image

from torch.utils.data import DataLoader
#DataLoader : 데이터를 batch 단위로 나눠서 처리(메모리 절약 + 빠름)
from PIL import ImageOps

import matplotlib.pyplot as plt


# 이미지 데이터를 텐서로 변환
from torchvision.datasets import EMNIST

transform = transforms.Compose([
transforms.Lambda(lambda img: ImageOps.invert(img)),              # 색 반전
    transforms.Lambda(lambda img: ImageOps.mirror(img)),          # 좌우 대칭 복원
    transforms.Lambda(lambda img: img.rotate(90, expand=True)),   # 시계방향 90도 회전
    transforms.ToTensor()
])
#MNIST는 기본적으 PIL 이미지(0~255) (Python Image Library)
#이미지 열기 (.jpg, .png, .bmp, .gif, .tiff, ...)
#크기 조절 (resize)
#잘라내기 (crop)
#회전, 흑백 변환, 필터 적용
#픽셀 값 접근
#TOTensor()를 쓰면 ->[0.0, 1.0]범위의 텐서로 변환


#==============================훈련/테스트 데이터셋=======================================#
# EMNIST 데이터셋 불러오기 (예: 'letters')
# train_dataset = EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
# test_dataset = EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
        return img, torch.tensor(label - 1, dtype=torch.long)

# ✅ 문제 핵심 요약
# PyTorch의 CrossEntropyLoss는 **라벨(label)**이 반드시
# torch.LongTensor 타입이어야 합니다.
#
# 하지만 현재 ABCDataset의 __getitem__()은 label - 1만 수행하고,
# 자료형(int, float)은 torch.Tensor인지 확실치 않습니다.


train_dataset = ABCDataset(train_dataset)
test_dataset = ABCDataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#======================================================================================#

#1. CNN 모델 만들기

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #pixel 데이터 추가
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, padding=0, bias=True)
        #합성곱 레이어

        self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max pooling
        #MaxPool2d(2, 2): 2x2 최대 풀링 → 크기를 절반으로 줄임


        self.conv2 = nn.Conv2d(3, 3, kernel_size=5, padding=0, bias=True)

        self.fc1 = nn.Linear(3 * 4 * 4, 3, bias=True)  # 완전연결,  # a,b,c 분류
        # self.fc2 = nn.Linear(32, 3)  # a,b,c 분류

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 3 * 4 * 4)  # Flatten
        x = x.view(x.size(0), -1)
        ##soft max한거
        x = self.fc1(x)

        return x

#forward: 입력이 모델을 통과할 때의 연산 정의
#ReLU: 비선형 활성화 함수 → 딥러닝에서 매우 중요
#view: 텐서를 펼쳐서 FC 레이어에 넣음



################################################## 주석처리 구분 ##################################################
######### 손실함수와 옵티마이저 정의 ##########
# model = CNN()
#
# # device : 모델과 데이터를 CPU, GPU 어디에 올릴지 정함
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # model을 옮기는 코드
# model.to(device)
# import torch.optim as optim
# # 예측 정갑 간의 차이 계산하는 함수
# criterion = nn.CrossEntropyLoss()
# # 옵티마이저는 모델의 weight를 업데이트
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# # 훈련 루프
# epochs = 5
#
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#
#     for images, labels in train_loader:
#         images, labels = images.to(device), (labels).to(device)
#
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
# ########################모델 정확도 평가##############################
# model.eval()
# correct = 0
# total = 0
#
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), (labels).to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f"Test Accuracy: {100 * correct / total:.2f}%")
#
#
# # 1단계 훈련한 모델 저장하는 코드
# # "mnist_cnn.pth" 파일이 생성됨 (이게 모델의 weight 저장본이에요
# torch.save(model.state_dict(), "mnist_cnn.pth")
################################################## 주석처리 구분 ##################################################




#######저장된 모델 불러오기#########
# model = CNN()
# model.load_state_dict(torch.load("mnist_cnn.pth"))
# model.eval()
#################################



#############################################################
model = CNN()


# Conv1 weights and bias
model.conv1.weight.data = torch.tensor([
    [[[12,13,33,20,42],[11,24,54,33,28],[45,51,19,8,22],[-14,14,-22,-20,-8],[-37,-21,-22,-42,-50]]],
    [[[23,21,-11,25,26],[27,4,16,-1,26],[2,20,-3,9,61],[44,8,35,31,15],[34,47,43,31,46]]],
    [[[1,0,-4,39,23],[0,9,33,47,51],[4,-15,20,41,-1],[-49,-37,17,-16,-7],[-52,-16,-48,-32,-19]]]
], dtype=torch.float32)
model.conv1.bias.data = torch.tensor([25, -41, 43], dtype=torch.float32)


# Conv2 weights and bias
model.conv2.weight.data = torch.tensor([[[[ 0.2592,  0.0712,  0.0715, -0.0580,  0.0023],
          [ 0.2311,  0.1724, -0.1054,  0.2066,  0.2696],
          [ 0.1158,  0.0411,  0.2065,  0.3031,  0.3592],
          [ 0.1499,  0.2090,  0.2459,  0.2897,  0.2585],
          [-0.0133,  0.0900,  0.0635, -0.0385, -0.0221]],
         [[ 0.1137,  0.1231,  0.1111,  0.1404,  0.2687],
          [ 0.1210, -0.0417,  0.1161,  0.2051,  0.1154],
          [-0.2439,  0.1009, -0.0055,  0.0110,  0.1335],
          [-0.1489, -0.0715, -0.1878, -0.1183, -0.0183],
          [-0.2312, -0.2451, -0.0377, -0.1718, -0.1398]],
         [[ 0.0295, -0.1335,  0.0305, -0.0602, -0.0554],
          [ 0.2743, -0.2187, -0.2796, -0.1247,  0.1965],
          [ 0.2387,  0.2164,  0.1097,  0.3045,  0.3437],
          [ 0.3677,  0.3251,  0.3557,  0.2930,  0.2443],
          [ 0.3595,  0.4071,  0.2222, -0.0350, -0.0430]]],
        [[[-0.1237, -0.2287, -0.2918, -0.0458, -0.1907],
          [-0.2162, -0.2087, -0.0187,  0.0290,  0.0375],
          [-0.0691, -0.0302, -0.0080, -0.0522,  0.1312],
          [ 0.0865,  0.0529, -0.1573, -0.0454,  0.0332],
          [ 0.1930,  0.0527,  0.1166, -0.0421,  0.1244]],
         [[-0.0041, -0.2357, -0.0787,  0.1704,  0.0813],
          [ 0.0986,  0.0840, -0.0510,  0.0084,  0.2955],
          [ 0.2652,  0.0935,  0.0479,  0.0579,  0.2364],
          [ 0.2606,  0.2075,  0.2360,  0.0723,  0.1834],
          [ 0.4022,  0.1996,  0.2317,  0.1236,  0.0708]],
         [[-0.2238,  0.0464, -0.0619, -0.1580, -0.4147],
          [-0.2716, -0.1588,  0.0247, -0.0309, -0.3717],
          [-0.3569, -0.3113, -0.0601,  0.0496,  0.0399],
          [-0.5184, -0.4376, -0.4133, -0.2760, -0.2177],
          [-0.3189, -0.3563, -0.3204, -0.3664, -0.0879]]],
        [[[-0.0726,  0.0422,  0.0626,  0.1008, -0.0765],
          [-0.0648, -0.1228, -0.0185,  0.0808,  0.0234],
          [-0.0309,  0.0880,  0.0239, -0.0979,  0.0564],
          [ 0.0055, -0.1021,  0.0080,  0.0100, -0.0107],
          [-0.0768, -0.1091, -0.1176,  0.0032, -0.0478]],
         [[ 0.0858,  0.0830, -0.0584, -0.0989,  0.0354],
          [-0.1199,  0.0478, -0.1170, -0.0730,  0.0468],
          [-0.0674, -0.0596, -0.0013, -0.0062, -0.0791],
          [-0.0726,  0.0252,  0.0043,  0.0175, -0.0742],
          [-0.0150, -0.0348, -0.0256,  0.0174, -0.1215]],
         [[ 0.0413,  0.0130,  0.0275, -0.1187, -0.0725],
          [-0.1053,  0.0121,  0.0139, -0.0512, -0.0700],
          [ 0.0062,  0.0559, -0.0301, -0.0066, -0.1207],
          [ 0.0778, -0.0234,  0.0898, -0.1014,  0.0780],
          [-0.0589,  0.0113,  0.0741, -0.0934, -0.0371]]]], dtype=torch.float32)
model.conv2.bias.data = torch.tensor([ 0.0135, -0.1800,  0.0046], dtype=torch.float32)


# FC1
model.fc1.weight.data = torch.tensor([
    [0.0358, -0.0892, -0.1154, 0.2058, 0.2404, 0.0259, -0.1440, 0.0059,
     0.2153, -0.1251, -0.1559, 0.0405, 0.1184, -0.1408, -0.2498, 0.0234,
     0.0066, -0.0768, 0.0770, 0.0476, 0.0091, 0.1527, -0.1512, -0.0963,
     -0.1414, 0.0247, -0.0792, -0.2076, 0.2913, 0.0852, 0.2078, -0.0901,
     0.1050, -0.0120, -0.0286, 0.0388, -0.1076, 0.1021, -0.1197, -0.0495,
     0.0179, 0.0120, -0.1090, -0.1356, 0.0408, -0.0453, 0.0429, 0.0946],

    [-0.1567,  0.1298,  0.2034,  0.0122, -0.0825, -0.0784,  0.1906,  0.1770,
         -0.0039, -0.1989,  0.0416, -0.1426,  0.0703,  0.0728,  0.1167, -0.1333,
          0.0852,  0.0266,  0.1161, -0.0150,  0.1577, -0.1318, -0.1142, -0.0008,
          0.2083,  0.0231, -0.1867,  0.0559, -0.1664, -0.0917, -0.1456, -0.1681,
          0.0595,  0.1351, -0.0901,  0.0638, -0.1194,  0.0442,  0.0220,  0.0419,
          0.0706,  0.0397, -0.0872,  0.1021, -0.0406, -0.1114,  0.0385, -0.1103],

    [0.1582, -0.0269, -0.1086, -0.0754, -0.1537, -0.0654, -0.0743, -0.1402,
     -0.0544, 0.1673, 0.1356, 0.0997, -0.1909, -0.1407, -0.0803, 0.0386,
     0.0003, -0.0249, 0.0890, -0.0099, -0.2609, 0.1044, 0.2266, 0.1749,
     -0.0273, 0.0117, 0.1234, 0.1758, -0.0406, -0.1432, 0.0826, -0.1058,
     0.0456, -0.0198, -0.0950, 0.0574, -0.0639, -0.0173, 0.0196, -0.1012,
     0.1156, 0.0139, -0.0454, -0.1070, -0.0792, -0.1303, -0.0382, 0.1213]
], dtype=torch.float32)

model.fc1.bias.data = torch.tensor([11, -7, -15], dtype=torch.float32)

# 테스트 출력
print("Conv1 Filter 0:\n", model.conv1.weight.data[0])
print("FC1 Weights for Class 0:\n", model.fc1.weight.data[0])

#############################################################


# 2단계: 나중에 weight만 확인하고 싶을 때
# model = CNN()
# model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()
print(model.conv1.weight.shape)      # torch.Size([16, 1, 3, 3])
print(model.conv1.weight[0])         # 첫 번째 필터 보기

# 모델 conv1의 weight
print("=== Conv1 Weights ===")
print(model.conv1.weight.shape)  # => torch.Size([16, 1, 3, 3])
print(model.conv1.weight)        # => 실제 값 출력

print("=== Conv2 Weights ===")
print(model.conv2.weight.shape)  # => torch.Size([16, 1, 3, 3])
print(model.conv2.weight)        # => 실제 값 출력

print("\n=== FC1 Weights ===")
print(model.fc1.weight.shape)     # ex) torch.Size([128, 1568])
print(model.fc1.weight)

print("\n=== bias ===")
print("conv1 bias:", model.conv1.bias.data)
print("conv2 bias:", model.conv2.bias.data)
print("fc1 bias:", model.fc1.bias.data)


# print("\n=== FC2 Weights ===")
# print(model.fc2.weight.shape)     # ex) torch.Size([10, 128])
# print(model.fc2.weight)

#################### 내꺼 이미지 테스트##################
transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
def binarize(img, threshold=128):
    return img.point(lambda p: 255 if p > threshold else 0)


image_folder = "C:/Users/kccistc/Desktop/handwritebold/"
# 사람 수
num_people = 4

# 파일명 생성: a_1.png ~ c_3.png
file_names = [f"{chr(ch)}_{i}.png"
              for i in range(1, num_people + 1)
              for ch in range(ord('a'), ord('c') + 1)]


for file in file_names:
    img_path = image_folder + file
    img = Image.open(img_path)


    img = Image.open(img_path).convert("L")
    # 이미지를 8bit 흑백으로 변환

    # img = ImageOps.invert(img)
    # # 이미지를 색상 반전(흑백이여야 가능)

    # img = ImageOps.pad(img, (28, 28), centering=(0.5, 0.5), color = 0)
    #입력 이미지가 이것보다 작으면 패딩 및 가운데 정렬
    #크더라도 비율 유지하면서 축소하는데 이 때 빈공간 회색으로 채움
    
    # img = binarize(img, threshold=128)
    # # 이미지 이진화

    img_tensor = transform(img).unsqueeze(0)
    img_np = img_tensor.squeeze().numpy()  # [1, 28, 28] → [28, 28]

    plt.close('all')
    #imshow()는 흑백이어도 컬러로 보여줄려고 함
    plt.imshow(img_np, cmap="gray")
    plt.title(f"Resized Image: {file}")
    plt.axis("off")
    plt.show()
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    predicted_letter = chr(predicted.item() + ord('a'))

    print("Output logits:", output)
    print(f"{file} → Predicted digit: {predicted_letter}")
    print()

# for i in range(3):
#     image, label = train_dataset[i]  # label: 1~26 (a~z)
#     char_label = chr(label + ord('a'))  # ASCII 매핑
#
#     plt.close('all')
#     #imshow()는 흑백이어도 컬러로 보여줄려고 함
#     plt.imshow(image.squeeze(), cmap= "gray")
#     plt.title(f"Label: {char_label} (index: {label})")
#     plt.axis("off")
#     plt.show()
#     plt.close('all')
######################################################
print("end")


