
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



# 이미지 데이터를 텐서로 변환
from torchvision.datasets import EMNIST

transform = transforms.Compose([
transforms.Lambda(lambda img: ImageOps.invert(img)),              # 색 반전
    transforms.Lambda(lambda img: ImageOps.mirror(img)),          # 좌우 대칭 복원
    transforms.Lambda(lambda img: img.rotate(90, expand=True)),   # 시계방향 90도 회전
    transforms.ToTensor()
])


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

model = CNN()

import numpy as np
import matplotlib.pyplot as plt

# .mem 데이터 불러오기
values = [int(line.strip(), 16) for line in open("./mem_outputs/b_4.mem") if line.strip().startswith("0x")]
img_array = np.array(values, dtype=np.uint8).reshape(28, 28)

plt.imshow(img_array, cmap='gray')  # 밝을수록 흰색
plt.title("Loaded .mem Image")
plt.axis('off')
plt.show()
# Conv1 weights and bias
model.conv1.weight.data = torch.tensor([
    [[[12,13,33,20,42],[11,24,54,33,28],[45,51,19,8,22],[-14,14,-22,-20,-8],[-37,-21,-22,-42,-50]]],
    [[[23,21,-11,25,26],[27,4,16,-1,26],[2,20,-3,9,61],[44,8,35,31,15],[34,47,43,31,46]]],
    [[[1,0,-4,39,23],[0,9,33,47,51],[4,-15,20,41,-1],[-49,-37,17,-16,-7],[-52,-16,-48,-32,-19]]]
], dtype=torch.float32)
model.conv1.bias.data = torch.tensor([25, -41, 43], dtype=torch.float32)


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

# bias도 넣고 싶다면 예시:
model.conv2.bias.data = torch.tensor([2, -23, 1], dtype=torch.float32)  # 또는 정수 bias

model.fc1.weight.data = torch.tensor([
    [5, -11, -15, 26, 31, 3, -18, 1, 28, -16, -20, 5, 15, -18, -32, 3, 1, -10, 10, 6, 1, 20, -19, -12, -18, 3, -10, -27, 37, 11, 27, -12, 13, -2, -4, 5, -14, 13, -15, -6, 2, 2, -14, -17, 5, -6, 5, 12],
    [-20, 17, 26, 2, -11, -10, 24, 23, 0, -25, 5, -18, 9, 9, 15, -17, 11, 3, 15, -2, 20, -17, -15, 0, 27, 3, -24, 7, -21, -12, -19, -22, 8, 17, -12, 8, -15, 6, 3, 5, 9, 5, -11, 13, -5, -14, 5, -14],
    [20, -3, -14, -10, -20, -8, -10, -18, -7, 21, 17, 13, -24, -18, -10, 5, 0, -3, 11, -1, -33, 13, 29, 22, -3, 1, 16, 23, -5, -18, 11, -14, 6, -3, -12, 7, -8, -2, 3, -13, 15, 2, -6, -14, -10, -17, -5, 16],
], dtype=torch.float32)

model.fc1.bias.data = torch.tensor([11, -7, -15], dtype=torch.float32)  # 또는 정수 bias

# 3. Conv1 연산 결과 보기
import torch
import torch.nn.functional as F
import numpy as np

# ✅ 입력: 28x28 이미지, 모든 값이 1
# ✅ 0~255까지 반복되는 28x28 input feature map 생성
def create_cyclic_input(height=28, width=28):
    """
    0부터 255까지 값이 반복되는 feature map 생성
    28x28 = 784개 픽셀이므로, 784 ÷ 256 = 3번 완전 반복 + 나머지 16개
    """
    total_pixels = height * width  # 784

    # 0~255 패턴을 필요한 만큼 반복
    values = []
    for i in range(total_pixels):
        values.append(1)  # 0, 1, 2, ..., 255, 0, 1, 2, ...

    # numpy 배열로 변환 후 28x28로 reshape
    input_array = np.array(values, dtype=np.float32).reshape(height, width)

    # PyTorch tensor로 변환 (batch_size=1, channels=1 차원 추가)
    input_tensor = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0)
    # shape: [1, 1, 28, 28]

    return input_tensor


import re


def load_mem_file_to_tensor(mem_file_path, height=28, width=28):
    """
    .mem 파일을 읽어서 PyTorch tensor로 변환
    """
    values = []

    with open(mem_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue

        # 16진수 값 추출 (0x형식)
        hex_match = re.search(r'0x([0-9A-Fa-f]+)', line)
        if hex_match:
            hex_value = hex_match.group(1)
            decimal_value = int(hex_value, 16)
            values.append(decimal_value)

    # numpy 배열로 변환 후 reshape
    input_array = np.array(values, dtype=np.float32).reshape(height, width)

    # PyTorch tensor로 변환 (batch_size=1, channels=1 차원 추가)
    input_tensor = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0)

    return input_tensor

# ✅ 새로운 input tensor 생성
# input_tensor = create_cyclic_input(28, 28)
input_tensor = load_mem_file_to_tensor('../PythonProject2/mem_outputs/b_4.mem', 28, 28)

print("🔹 새로운 Input Feature Map 정보:")
print(f"Shape: {input_tensor.shape}")
print(f"Min value: {input_tensor.min().item()}")
print(f"Max value: {input_tensor.max().item()}")

# ✅ 이제 이 input_tensor로 CNN 연산 수행
print("\n" + "=" * 80)
print("🔹 새로운 Input으로 CNN 연산 시작")
print("=" * 80)

# ✅ Conv1 연산 - 전체 위치에 대한 상세 계산
print("\n" + "=" * 80)
print("🔹 새로운 Input으로 CNN 연산 시작 - 전체 위치 상세 계산")
print("=" * 80)

# input_tensor가 정의되어 있다고 가정
input_feature = input_tensor[0]  # shape: [3, 12, 12]
weight = model.conv1.weight.data  # weight shape 확인
bias = model.conv1.bias.data

# weight 텐서의 실제 shape 출력
print(f"🔍 Weight shape: {weight.shape}")
print(f"🔍 Input feature shape: {input_feature.shape}")
print(f"🔍 Bias shape: {bias.shape}")

# weight shape에 따라 input channel 수 결정
num_input_channels = weight.shape[1]  # weight의 두 번째 차원이 input channel 수
num_output_channels = weight.shape[0]  # weight의 첫 번째 차원이 output channel 수
kernel_h, kernel_w = weight.shape[2], weight.shape[3]

print(f"🔍 Input channels: {num_input_channels}, Output channels: {num_output_channels}")
print(f"🔍 Kernel size: {kernel_h}x{kernel_w}")

# 출력 크기 계산 (padding=0, stride=1 가정)
output_h = input_feature.shape[1] - kernel_h + 1
output_w = input_feature.shape[2] - kernel_w + 1
print(f"🔍 Output size: {output_h}x{output_w}")

# 실제 PyTorch conv1 연산 (비교용)
conv1_out = model.conv1(input_tensor)
print(f"🔍 실제 PyTorch 출력 shape: {conv1_out.shape}")

# 각 출력 채널에 대해 계산
for output_channel in range(min(1, num_output_channels)):  # 첫 번째 채널만 상세히 (너무 많으면 조정)
    print(f"\n" + "=" * 60)
    print(f"🎯 출력 채널 {output_channel} 상세 계산")
    print("=" * 60)

    # 각 출력 위치에 대해 계산
    for output_y in range(output_h):
        for output_x in range(output_w):
            print(f"\n[🔎 위치 ({output_y},{output_x}) 계산]")

            acc = 0
            channel_contributions = []  # 각 입력 채널의 기여도 저장

            # 각 입력 채널에 대해 컨볼루션 계산
            for in_c in range(num_input_channels):
                kernel = weight[output_channel, in_c]  # kernel shape: [H, W]

                # input feature의 해당 채널에서 패치 추출
                if in_c < input_feature.shape[0]:  # input channel이 존재하는지 확인
                    patch = input_feature[in_c, output_y:output_y + kernel_h, output_x:output_x + kernel_w]
                else:
                    print(f"⚠️ 입력 채널 {in_c}가 존재하지 않습니다.")
                    continue

                mul = patch * kernel  # element-wise 곱
                sum_mul = mul.sum().item()
                channel_contributions.append(sum_mul)

                print(f"  📝 입력 채널 {in_c}: patch×kernel 합계 = {sum_mul:.4f}")

                # 상세 정보 출력 (선택적으로 - 너무 많으면 주석 처리)
                if output_y < 2 and output_x < 2:  # 처음 몇 개만 상세히
                    print(f"    입력 패치:\n{patch}")
                    print(f"    커널:\n{kernel}")
                    print(f"    곱:\n{mul}")

                acc += sum_mul

            # bias 더하기 전 결과 상세 분석
            print(f"\n  🧮 각 채널별 기여도:")
            for i, contrib in enumerate(channel_contributions):
                print(f"    채널 {i}: {contrib:8.4f}")

            print(f"  📊 전체 채널 합계 (bias 전): {acc:.4f}")
            print(f"  🎯 Bias 값: {bias[output_channel].item():.4f}")

            # bias 더함
            final_result = acc + bias[output_channel].item()

            # 결과 비교
            pytorch_result = conv1_out[0, output_channel, output_y, output_x].item()
            print(f"  ✅ 최종 결과 (bias 후): {final_result:.4f}")
            print(f"  🔍 PyTorch 결과: {pytorch_result:.4f}")
            print(f"  📊 차이: {abs(final_result - pytorch_result):.6f}")

            # bias 전후 비교
            print(f"  📈 Bias 효과: {acc:.4f} → {final_result:.4f} (변화: {final_result - acc:+.4f})")

            if in_c < input_feature.shape[0]:  # input channel이 존재하는지 확인
                patch = input_feature[in_c, output_y:output_y + kernel_h, output_x:output_x + kernel_w]
            else:
                print(f"⚠️ 입력 채널 {in_c}가 존재하지 않습니다.")
                continue

            mul = patch * kernel  # element-wise 곱
            sum_mul = mul.sum().item()

            print(f"  📝 입력 채널 {in_c}: patch×kernel 합계 = {sum_mul:.1f}")

            # 상세 정보 출력 (선택적으로 - 너무 많으면 주석 처리)
            if output_y < 2 and output_x < 2:  # 처음 몇 개만 상세히
                print(f"    입력 패치:\n{patch}")
                print(f"    커널:\n{kernel}")
                print(f"    곱:\n{mul}")

            acc += sum_mul

        # bias 더함
        acc += bias[output_channel].item()

        # 결과 비교
        pytorch_result = conv1_out[0, output_channel, output_y, output_x].item()
        print(f"  ✅ 계산 결과: {acc:.1f}")
        print(f"  🔍 PyTorch 결과: {pytorch_result:.1f}")
        print(f"  📊 차이: {abs(acc - pytorch_result):.6f}")

print(f"\n" + "=" * 80)
print("🎉 전체 컨볼루션 연산 상세 계산 완료!")
print("=" * 80)

print(f"➡️ 실제 conv1_out 값: {conv1_out[0, output_channel, output_y, output_x].item()}")
print("🔹 [Conv1 출력] shape:", conv1_out.shape)

# Conv1 출력의 모든 위치 값 출력
print(f"\n🔹 Conv1 출력 상세 (모든 위치):")
print(f"출력 shape: {conv1_out.shape}")

for channel in range(conv1_out.shape[1]):  # 각 출력 채널
    print(f"\n📍 Conv1 채널 {channel}:")
    channel_output = conv1_out[0, channel]  # [H, W]

    # 각 위치별 값 출력
    for y in range(channel_output.shape[0]):
        row_values = []
        for x in range(channel_output.shape[1]):
            row_values.append(f"{channel_output[y, x].item():8.1f}")
        print(f"  y={y}: [" + ", ".join(row_values) + "]")

# 간단한 요약도 출력
print(f"\n📊 Conv1 출력 요약:")
for i in range(conv1_out.shape[1]):
    channel_data = conv1_out[0, i]
    print(
        f"채널 {i}: min={channel_data.min().item():.1f}, max={channel_data.max().item():.1f}, mean={channel_data.mean().item():.1f}")
# ✅ ReLU 적용
relu1_out = F.relu(conv1_out)
print("\n" + "=" * 60)
print("🔹 [ReLU1 출력] (Conv1 → ReLU) - 행렬 형식")
print("=" * 60)

for channel in range(relu1_out.shape[1]):
    print(f"\n🔹 ReLU1 채널 {channel}:")
    channel_output = relu1_out[0, channel]  # [H, W]

    # 행렬 형식으로 출력
    for y in range(channel_output.shape[0]):
        row_values = []
        for x in range(channel_output.shape[1]):
            row_values.append(f"{channel_output[y, x].item():8.1f}")
        print("  [" + ", ".join(row_values) + "]")


# ✅ MaxPooling1 적용
pool1_out = F.max_pool2d(relu1_out, 2, 2)
print("\n🔹 [MaxPool1 출력] shape:", pool1_out.shape)
for i in range(pool1_out.shape[1]):
    print(f"MaxPool1 채널 {i} 값:")
    print(pool1_out[0, i])

# ✅ MaxPool1과 Conv2 Weight 곱셈 상세 출력
print("\n" + "=" * 80)
print("🔹 MaxPool1과 Conv2 Weight 곱셈 상세 과정")
print("=" * 80)

# Conv2의 weight shape: [out_channels, in_channels, kernel_height, kernel_width]
conv2_weight = model.conv2.weight.data  # shape: [3, 3, 5, 5]
conv2_bias = model.conv2.bias.data  # shape: [3]

# MaxPool1 출력 shape: [1, 3, 12, 12]
pool1_data = pool1_out[0]  # batch dimension 제거 -> [3, 12, 12]

print(f"MaxPool1 출력 shape: {pool1_data.shape}")
print(f"Conv2 weight shape: {conv2_weight.shape}")

# 실제 Conv2 연산 결과
actual_conv2_out = model.conv2(pool1_out)
print("실제 Conv2 출력:")
for i in range(actual_conv2_out.shape[1]):
    print(f"채널 {i}:")
    print(actual_conv2_out[0, i])

# ✅ Conv2 연산
conv2_out = model.conv2(pool1_out)
print("\n🔹 [Conv2 출력] shape:", conv2_out.shape)
for i in range(conv2_out.shape[1]):
    print(f"Conv2 채널 {i} 값:")
    print(conv2_out[0, i])

# pool1_out: [1, 3, 12, 12]
input_feature = pool1_out[0]  # shape: [3, 12, 12]
weight = model.conv2.weight.data  # shape: [3, 3, 5, 5]
bias = model.conv2.bias.data  # shape: [3]

output_channel = 0
output_y = 0
output_x = 0

acc = 0
print(f"\n[🔎 conv2 계산 상세] 채널 {output_channel} / 위치 ({output_y},{output_x})")

for in_c in range(3):  # 3 input channels
    kernel = weight[output_channel, in_c]  # [5x5]
    patch = input_feature[in_c, output_y:output_y + 5, output_x:output_x + 5]  # [5x5]
    mul = patch * kernel  # element-wise 곱
    sum_mul = mul.sum().item()

    print(f"\n🧩 입력 채널 {in_c}의 patch × kernel:")
    print("입력 패치:")
    print(patch)
    print("커널:")
    print(kernel)
    print("곱:")
    print(mul)
    print(f"합계: {sum_mul}")

    acc += sum_mul

# bias 더함
acc += bias[output_channel].item()
print(f"\n✅ Bias({bias[output_channel].item()}) 더한 최종 결과: {acc}")
print(f"➡️ 실제 conv2_out 값: {conv2_out[0, output_channel, output_y, output_x].item()}")

# ✅ ReLU2 적용
relu2_out = F.relu(conv2_out)
print("\n🔹 [ReLU2 출력] (Conv2 → ReLU)")
for i in range(relu2_out.shape[1]):
    print(f"ReLU2 채널 {i} 값:")
    print(relu2_out[0, i])

# ✅ MaxPooling2 적용
pool2_out = F.max_pool2d(relu2_out, 2, 2)
print("\n🔹 [MaxPool2 출력] shape:", pool2_out.shape)
for i in range(pool2_out.shape[1]):
    print(f"MaxPool2 채널 {i} 값:")
    print(pool2_out[0, i])

# ✅ Flatten (벡터화)
flatten_out = pool2_out.view(pool2_out.size(0), -1)
print("\n🔹 [Flatten 출력] shape:", flatten_out.shape)
print("Flatten 값:")
print(flatten_out[0])  # 배치 크기 1이므로 [0]만 출력

# ✅ FC1 연산 (Fully Connected Layer)
fc1_out = model.fc1(flatten_out)

# ✅ FC1 연산 수동 계산 상세 출력
print("\n" + "=" * 80)
print("🔍 [FC1 수동 계산 상세 출력]")
print("=" * 80)

fc1_weight = model.fc1.weight.data  # shape: [3, 48]
fc1_bias = model.fc1.bias.data      # shape: [3]
flatten_input = flatten_out[0]      # shape: [48]

for class_idx in range(3):  # 출력 클래스 3개
    print(f"\n🎯 클래스 {class_idx} 계산:")
    acc = 0
    contribs = []

    for i in range(flatten_input.shape[0]):
        input_val = flatten_input[i].item()
        weight_val = fc1_weight[class_idx][i].item()
        prod = input_val * weight_val
        acc += prod
        contribs.append((i, input_val, weight_val, prod))

    # 출력 상세
    for idx, inp, w, p in contribs:
        print(f"  입력[{idx:02d}] × 가중치[{idx:02d}] = {inp:.1f} × {w:.1f} = {p:.1f}")

    print(f"  📊 가중치곱 합계 (bias 전): {acc:.1f}")
    print(f"  🎁 bias: {fc1_bias[class_idx].item():.1f}")
    acc += fc1_bias[class_idx].item()
    print(f"  ✅ 최종 결과: {acc:.1f}")
    print(f"  🔍 PyTorch 출력 결과: {fc1_out[0][class_idx].item():.1f}")
    print(f"  📉 차이: {abs(acc - fc1_out[0][class_idx].item()):.6f}")

print("\n🔹 [FC1 출력] (최종 출력 벡터 - 클래스 3개)")
print(fc1_out[0])  # [0]: 첫 번째 배치에 대한 예측 결과
