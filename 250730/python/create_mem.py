# # ***************************************************************** #
# #  .png파일을
# #  1. gray
# #  2. r,g,b
# #  3. normalized_gray
# #  scale해서 mem파일로 만들어줌(hexa)
# #  아래에서 각자의 경로 설정
# #  image_path : .png파일이 저장된 위치
# #  mem_path   : mem파일 저장할 위치
# # ***************************************************************** #

import os
from PIL import Image
import numpy as np
image_path = r'C:\github\Braille_generator_FPGA\handwritebold'
mem_path = r'C:\github\Braille_generator_FPGA\mem_outputs_dongeun'



# normalize없이
def convert_images_to_gray_mem(image_folder, mem_folder, resize=(28, 28)):
    os.makedirs(mem_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            base_name = os.path.splitext(filename)[0]
            mem_filename = f"{base_name}_gray.mem"
            mem_path = os.path.join(mem_folder, mem_filename)

            img = Image.open(image_path).convert('L')
            img = img.resize(resize)
            img_array = np.array(img, dtype=np.uint8).flatten()

            with open(mem_path, 'w') as f:
                for val in img_array:
                    f.write(f"0x{val:02x}\n")

            print(f"[GRAY] Saved: {mem_filename}")
            
            
            
#normalize 한 경우            
def convert_images_to_gray_normalized_mem(image_folder, mem_folder, resize=(28, 28)):
    os.makedirs(mem_folder, exist_ok=True)

    mean = 0.1307
    std = 0.3081

    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            base_name = os.path.splitext(filename)[0]
            mem_filename = f"{base_name}_gray_norm.mem"
            mem_path = os.path.join(mem_folder, mem_filename)

            # Load and preprocess
            img = Image.open(image_path).convert('L')
            img = img.resize(resize)
            img_array = np.array(img, dtype=np.uint8).flatten()

            # Normalize and scale
            norm_array = (img_array / 255.0 - mean) / std
            scaled_array = np.clip(np.round(norm_array * 128), -128, 127).astype(np.int8)

            # Save as signed hex
            with open(mem_path, 'w') as f:
                for val in scaled_array:
                    # 8-bit signed to 2's complement hex
                    hex_val = int(val) & 0xFF
                    f.write(f"0x{hex_val:02X}\n")

            print(f"[GRAY+NORM] Saved: {mem_filename}")
            
            
            
            
            
# r,g,b 변환 normalize없이
def convert_images_to_rgb_mem(image_folder, mem_folder, resize=(28, 28)):
    os.makedirs(mem_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            base_name = os.path.splitext(filename)[0]
            mem_filename = f"{base_name}_rgb.mem"
            mem_path = os.path.join(mem_folder, mem_filename)

            img = Image.open(image_path).convert('RGB')
            img = img.resize(resize)
            img_array = np.array(img).reshape(-1, 3)

            with open(mem_path, 'w') as f:
                for r, g, b in img_array:
                    f.write(f"0x{r:02x}{g:02x}{b:02x}\n")

            print(f"[RGB] Saved: {mem_filename}")

convert_images_to_gray_normalized_mem(image_path, mem_path)
convert_images_to_gray_mem(image_path, mem_path)
convert_images_to_rgb_mem(image_path, mem_path)