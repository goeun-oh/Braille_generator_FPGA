import torch
import numpy as np
import random


def create_random_input_for_verilog(height=28, width=28, max_value=15, seed=None):
    """
    0~max_value 범위의 랜덤 input feature map 생성

    Args:
        height: feature map 높이
        width: feature map 너비
        max_value: 최대값 (0~max_value 범위)
        seed: 랜덤 시드 (재현 가능한 결과를 위해)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    total_pixels = height * width

    # 0~max_value 범위의 랜덤 값 생성
    values = np.random.randint(0, max_value + 1, size=total_pixels, dtype=np.uint8)

    # 2D 배열로 reshape
    input_2d = values.reshape(height, width)
    return input_2d


def generate_mem_files_random(height=28, width=28, max_value=15, seed=42):
    """
    Verilog 검증용 랜덤 .mem 파일들 생성

    Args:
        height: feature map 높이
        width: feature map 너비
        max_value: 픽셀 최대값 (0~max_value)
        seed: 랜덤 시드
    """
    # 랜덤 Input feature map 생성
    input_map = create_random_input_for_verilog(height, width, max_value, seed)

    print(f"🔹 랜덤 .mem 파일 생성 중... (크기: {height}x{width}, 최대값: {max_value})")
    print(f"🎲 랜덤 시드: {seed}")

    # 생성된 데이터 통계 출력
    print(f"📊 생성된 값 분포:")
    print(f"   최솟값: {input_map.min()}")
    print(f"   최댓값: {input_map.max()}")
    print(f"   평균값: {input_map.mean():.2f}")
    print(f"   총 픽셀 수: {input_map.size}")

    # 값 분포 히스토그램
    unique, counts = np.unique(input_map, return_counts=True)
    print(f"📈 값별 개수:")
    for val, count in zip(unique, counts):
        percentage = (count / input_map.size) * 100
        print(f"   값 {val:2d}: {count:3d}개 ({percentage:5.1f}%)")

    # 1. Input Feature Map (.mem 파일)
    filename = f'input_feature_map_random_{height}x{width}_max{max_value}.mem'
    with open(f'../PythonProject2/{filename}', 'w') as f:
        f.write(f"// Random Input Feature Map ({height}x{width})\n")
        f.write(f"// Format: 8-bit unsigned integer (0-{max_value})\n")
        f.write(f"// Total: {height * width} values\n")
        f.write(f"// Random seed: {seed}\n")
        f.write(f"// Address mapping: addr = row*{width} + col\n\n")

        addr = 0
        for row in range(height):
            for col in range(width):
                value = input_map[row, col]
                f.write(f"0x{value:02X}  // [{row:2d},{col:2d}] = {value:3d}\n")
                addr += 1

    print(f"✅ 파일 생성 완료: {filename}")

    # 처음 몇 행 미리보기
    print(f"\n🔍 생성된 feature map 미리보기 (처음 5x5):")
    preview_size = min(5, height, width)
    for i in range(preview_size):
        row_str = " ".join([f"{input_map[i, j]:2d}" for j in range(preview_size)])
        if width > preview_size:
            row_str += " ..."
        print(f"   [{i}]: {row_str}")
    if height > preview_size:
        print("   ...")

    return input_map


# 다양한 설정으로 생성 예시
if __name__ == "__main__":
    # 기본 설정 (28x28, 최대값 15)
    fmap1 = generate_mem_files_random(height=28, width=28, max_value=15, seed=42)

    print("\n" + "=" * 60)

    # 작은 크기 테스트용 (12x12, 최대값 15)
    fmap2 = generate_mem_files_random(height=12, width=12, max_value=15, seed=123)

    print("\n" + "=" * 60)

    # 다른 시드로 생성
    fmap3 = generate_mem_files_random(height=28, width=28, max_value=15, seed=999)