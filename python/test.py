import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ✅ 입력: 1로 채워진 28x28 텐서
input_tensor = torch.ones((1, 1, 28, 28), dtype=torch.float32)
model.conv1.weight.data = torch.tensor([
    [[[12,13,33,20,42],[11,24,54,33,28],[45,51,19,8,22],[-14,14,-22,-20,-8],[-37,-21,-22,-42,-50]]],
    [[[23,21,-11,25,26],[27,4,16,-1,26],[2,20,-3,9,61],[44,8,35,31,15],[34,47,43,31,46]]],
    [[[1,0,-4,39,23],[0,9,33,47,51],[4,-15,20,41,-1],[-49,-37,17,-16,-7],[-52,-16,-48,-32,-19]]]
], dtype=torch.float32)
model.conv1.bias.data = torch.tensor([25, -41, 43], dtype=torch.float32)


# ✅ 필터 0 추출 (1 input channel, 1 output channel, 5x5 커널)
filter0 = model.conv1.weight.data[0].unsqueeze(0)  # shape: [1, 1, 5, 5]
bias0 = model.conv1.bias.data[0].unsqueeze(0)      # shape: [1]

# ✅ Convolution 수행
with torch.no_grad():
    conv_result = F.conv2d(input_tensor, weight=filter0, bias=bias0, stride=1, padding=0)

# ✅ 출력 확인
print("=== Conv1 Filter 0 연산 결과 ===")
print(conv_result.shape)  # [1, 1, 24, 24]
print(conv_result)

# ✅ 시각화
plt.imshow(conv_result.squeeze().numpy(), cmap='gray')
plt.title("Conv1 Filter 0 Output")
plt.axis("off")
plt.show()
