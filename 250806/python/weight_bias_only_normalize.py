# ********************* a. CNN 모델  ************************ #
# ********************************************************** #

import torch.nn as nn
import torch.nn.functional as F
import torch

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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 3 * 4 * 4)  # Flatten
        x = x.view(x.size(0), -1)
        ##soft max한거
        x = self.fc1(x)

        return x
# forward: 입력이 모델을 통과할 때의 연산 정의
# ReLU: 비선형 활성화 함수 → 딥러닝에서 매우 중요
# view: 텐서를 펼쳐서 FC 레이어에 넣음
# *************************************************************** #
# *************************************************************** #





# # *************** b. 저장된 모델 weight, bias 확인 **************** #
# # ************************************************************** #

model = CNN(3, 6)
model.load_state_dict(torch.load("cnn_c3_6_ep5_a_z.pth"))
model.eval()

# 모델 conv1의 weight
print("\n=== Conv1 Weights ===")
# print(model.conv1.weight.shape)  # => torch.Size([16, 1, 3, 3])
# print(model.conv1.weight)  # => 실제 값 출력
# conv1 weight 전체 텐서 보기
weights = model.conv1.weight.data
# 최솟값과 최댓값 출력
print("Conv1 weight min:", weights.min().item())
print("Conv1 weight max:", weights.max().item())

print("\n=== Conv2 Weights ===")
# print(model.conv2.weight.shape)  # => torch.Size([16, 1, 3, 3])
# print(model.conv2.weight)  # => 실제 값 출력
weights = model.conv2.weight.data
# 최솟값과 최댓값 출력
print("Conv2 weight min:", weights.min().item())
print("Conv2 weight max:", weights.max().item())

print("\n=== FC1 Weights ===")
# print(model.fc1.weight.shape)  # ex) torch.Size([128, 1568])
# print(model.fc1.weight)
weights = model.fc1.weight.data
# 최솟값과 최댓값 출력
print("fc1 weight min:", weights.min().item())
print("fc1 weight max:", weights.max().item())



print("\nConv1 bias:")
# print(model.conv1.bias)  # => 실제 값 출력
bias = model.conv1.bias.data
# 최솟값과 최댓값 출력
print("Conv1 bias min:", bias.min().item())
print("Conv1 bias max:", bias.max().item())


print("\nConv2 bias:")
# print(model.conv2.bias)  # => 실제 값 출력
bias = model.conv2.bias.data
# 최솟값과 최댓값 출력
print("Conv2 bias min:", bias.min().item())
print("Conv2 bias max:", bias.max().item())


print("\nfc1 bias:")
# print(model.fc1.bias)  # => 실제 값 출력
bias = model.fc1.bias.data
# 최솟값과 최댓값 출력
print("fc1 bias min:", bias.min().item())
print("fc1 bias max:", bias.max().item())
# ************************************************************** #
# ************************************************************** #




# # **************** c. weight, bias mem파일로 변환 **************** #
# # ************************************************************** #
import os

print("\n================== bias, weight normalize ==================")


weight_bias_path = r'C:\github\Braille_generator_FPGA\weight_biaz_normalize'

scale = 128  # 보통 -1.0 ~ 1.0 사이의 값이면 128 곱해서 int8 사용
scale2 = 64
max_val = 32767
min_val = -32768

# conv1.weight
weights = model.conv1.weight.data.clone().cpu()
print("\nconv1_weights:", weights)
print("\nconv1_weights.shape:", weights.shape)
int_weights = (weights * scale).clamp(min_val, max_val).to(torch.int32)
print("conv1_int_weights.shape:", int_weights.shape)
# Convert to nested list
nested_list = int_weights.tolist()
# Optional: Save as .py-style format
save_path = os.path.join(weight_bias_path, "conv1_weight_decimal.mem")
with open(save_path, "w") as f:
    f.write("conv1_weights = [\n")
    for out_ch in nested_list:
        f.write("    [\n")
        for in_ch in out_ch:
            f.write("        [\n")
            for row in in_ch:
                f.write(f"            {row},\n")
                # formatted_row = ", ".join(f"{v:.4f}" for v in row)
                # f.write(f"            [{formatted_row}],\n")
            f.write("        ],\n")
        f.write("    ],\n")
    f.write("]\n")
print("✅ conv1_weights_decimal_write_done")
# with open(os.path.join(weight_bias_path, "conv1_weight.mem"), "w") as f:
#     for out_ch in range(int_weights.shape[0]):        # 16
#         for in_ch in range(int_weights.shape[1]):     # 1
#             for i in range(int_weights.shape[2]):     # 3
#                 for j in range(int_weights.shape[3]): # 3
#                     val = int_weights[out_ch][in_ch][i][j].item()
#                     hex_val = f"{(val & 0xFF):02x}"  # 2-digit hex (8bit signed)
#                     f.write(f"0x{hex_val}\n")
# print("conv1_weights_write_done")
#


# conv2.weight
weights = model.conv2.weight.data.clone().cpu()
print("\nconv2_weights:", weights)
print("\nconv2_weights.shape:", weights.shape)
int_weights = (weights * scale).clamp(min_val, max_val).to(torch.int32)
print("conv2_int_weights.shape:", int_weights.shape)
# Python 리스트로 변환
nested_list = int_weights.tolist()
# 저장 경로 설정
save_path = os.path.join(weight_bias_path, "conv2_weight_decimal.mem")
# Python 리스트 형태로 저장
with open(save_path, "w") as f:
    f.write("conv2_weights = [\n")
    for out_ch in nested_list:
        f.write("    [\n")
        for in_ch in out_ch:
            f.write("        [\n")
            for row in in_ch:
                f.write(f"            {row},\n")
                # formatted_row = ", ".join(f"{v:.4f}" for v in row)
                # f.write(f"            [{formatted_row}],\n")
            f.write("        ],\n")
        f.write("    ],\n")
    f.write("]\n")
print("✅ conv2_weights_decimal_write_done")
# with open(os.path.join(weight_bias_path, "conv2_weight.mem"), "w") as f:
#     for out_ch in range(int_weights.shape[0]):
#         for in_ch in range(int_weights.shape[1]):
#             for i in range(int_weights.shape[2]):
#                 for j in range(int_weights.shape[3]):
#                     val = int_weights[out_ch][in_ch][i][j].item()
#                     hex_val = f"{(val & 0xFF):02x}"  # 2-digit hex (8bit signed)
#                     f.write(f"0x{hex_val}\n")
# print("conv2_weights_write_done")



# fc1.weight
weights = model.fc1.weight.data.clone().cpu()
print("\nfc1_weights:", weights)
print("\nfc1_weights.shape:", weights.shape)
int_weights = (weights * scale2).clamp(min_val, max_val).to(torch.int32)
print("fc1_int_weights.shape:", int_weights.shape)
# Python 리스트로 변환
nested_list = int_weights.tolist()
# 저장 경로 설정
save_path = os.path.join(weight_bias_path, "fc1_weight_decimal.mem")
# Python 리스트 형태로 저장
with open(save_path, "w") as f:
    f.write("fc1_weights = [\n")
    for out_ch in nested_list:
        f.write(f"    {out_ch},\n")
        # formatted_line = ", ".join(f"{v:.4f}" for v in out_ch)
        # f.write(f"    [{formatted_line}],\n")
    f.write("]\n")
print("✅ fc1_weights_decimal_write_done")
# with open(os.path.join(weight_bias_path, "fc1_weight.mem"), "w") as f:
#     for out_ch in range(int_weights.shape[0]):
#         for in_ch in range(int_weights.shape[1]):
#             val = int_weights[out_ch][in_ch].item()
#             hex_val = f"{(val & 0xFF):02x}"  # 2-digit hex (8bit signed)
#             f.write(f"0x{hex_val}\n")
# print("fc1_weights_write_done")



# 경로 설정
save_path1 = os.path.join(weight_bias_path, "conv1_bias_decimal.mem")
save_path2 = os.path.join(weight_bias_path, "conv2_bias_decimal.mem")
save_path3 = os.path.join(weight_bias_path, "fc1_bias_decimal.mem")

# ---------- conv1.bias ----------
bias = model.conv1.bias.data.clone().cpu()
print("\nconv1_bias:", bias)
print("\nconv1_bias.shape:", bias.shape)
int_bias = (bias * scale).clamp(min_val, max_val).to(torch.int32)
print("conv1_int_bias.shape:", int_bias.shape)

with open(save_path1, "w") as f:
    f.write("conv1_bias = [")
    for val in int_bias.tolist():
        f.write(f"{val}, ")
        # f.write(f"{val:.4f}, ")
    f.write("]\n")
print("✅ conv1_bias_decimal_write_done")

# ---------- conv2.bias ----------
bias = model.conv2.bias.data.clone().cpu()
print("\nconv2_bias:", bias)
print("\nconv2_bias.shape:", bias.shape)
int_bias = (bias * scale).clamp(min_val, max_val).to(torch.int32)
print("conv2_int_bias.shape:", int_bias.shape)

with open(save_path2, "w") as f:
    f.write("conv2_bias = [")
    for val in int_bias.tolist():
        f.write(f"{val}, ")
        # f.write(f"{val:.4f}, ")
    f.write("]\n")
print("✅ conv2_bias_decimal_write_done")

# ---------- fc1.bias ----------
bias = model.fc1.bias.data.clone().cpu()
print("\nfc1_bias:", bias)
print("\nfc1_bias.shape:", bias.shape)
int_bias = (bias * scale2).clamp(min_val,max_val).to(torch.int32)
print("fc1_int_bias.shape:", int_bias.shape)

with open(save_path3, "w") as f:
    f.write("fc1_bias = [")
    for val in int_bias.tolist():
        f.write(f"{val}, ")
        # f.write(f"{val:.4f}, ")
    f.write("]\n")
print("✅ fc1_bias_decimal_write_done")


