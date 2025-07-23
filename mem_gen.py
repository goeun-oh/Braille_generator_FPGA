with open("input_fmap.mem", "w") as f:
    for i in range(784):  # 28 x 28
        f.write(f"{1:#04x}\n")  # 0x00 ~ 0x30f
