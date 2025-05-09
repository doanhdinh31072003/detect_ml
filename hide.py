def hide_bits_with_commas(text, bitstring):
    lines = text.split('\n')
    result = []
    bit_index = 0

    for line in lines:
        if bit_index >= len(bitstring):
            result.append(line)
            continue


        if " và " in line or ", và " in line:
            bit = bitstring[bit_index]
            bit_index += 1

            if bit == '1':
          
                line = line.replace(" và ", ", và ", 1)
            elif bit == '0':
   
                line = line.replace(", và ", " và ", 1)

        result.append(line)

    return '\n'.join(result)


print("Nhập văn bản cần giấu (kết thúc bằng dòng trống):")
lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)

text = '\n'.join(lines)

bitstring = input("Nhập chuỗi bit cần giấu (chỉ gồm 0 và 1): ")


hidden_text = hide_bits_with_commas(text, bitstring)


print("\n📄 Văn bản sau khi giấu tin:\n")
print(hidden_text)
# Tôi ăn bánh mì, bơ và sữa mỗi sáng.
# Mẹ tôi nấu cơm, cá và rau.
# Tôi ăn bánh mì, bơ và sữa mỗi sáng.
# Mẹ tôi nấu cơm, cá và rau.
