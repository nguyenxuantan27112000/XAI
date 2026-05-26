import pandas as pd

# ==========================================
# BƯỚC 0: TẢI DỮ LIỆU Y TẾ (BỆNH TIỂU ĐƯỜNG)
# ==========================================
print("Đang tải dữ liệu Pima Indians Diabetes...")
# Sử dụng link raw csv công khai để chạy được ngay trên Colab
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df_full = pd.read_csv(url)

# Chọn 4 đặc trưng quan trọng nhất để luật dễ đọc + Cột Kết quả
features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
target = 'Outcome'
df = df_full[features + [target]].copy()

# ==========================================
# BƯỚC 1 & 2: NGÔN NGỮ HÓA THEO KHOẢNG ĐSGT
# ==========================================
# Chuẩn hóa Min-Max đưa toàn bộ dữ liệu về không gian [0, 1]
for col in features:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Giả lập Khoảng định lượng ngữ nghĩa Phi(x) cho Y tế
def linguistic_mapping(val):
    if 0 <= val <= 0.35:
        return "Thấp"
    elif 0.35 < val <= 0.65:
        return "Bình thường" # Đổi chữ "Trung bình" thành "Bình thường" cho hợp y tế
    else:
        return "Cao"

# Áp dụng ngôn ngữ hóa
ling_cols = []
for col in features:
    ling_col_name = col + '_NgonNgu'
    df[ling_col_name] = df[col].apply(linguistic_mapping)
    ling_cols.append(ling_col_name)

# ==========================================
# BƯỚC 3: SINH LUẬT & GIẢI QUYẾT XUNG ĐỘT
# ==========================================
print("Đang phân hoạch ngữ nghĩa và trích xuất luật...")
rules_df = df.groupby(ling_cols + [target]).size().reset_index(name='Tan_suat')

rule_base = {}
for index, row in rules_df.iterrows():
    antecedent = tuple(row[ling_cols])
    # Chuyển đổi nhãn 0/1 thành ngôn ngữ
    consequent = "CÓ nguy cơ Tiểu đường" if row[target] == 1 else "KHÔNG bị Tiểu đường"
    frequency = row['Tan_suat']
    
    if antecedent not in rule_base:
        rule_base[antecedent] = {'Ket_luan': consequent, 'Tan_suat': frequency}
    else:
        # Giải quyết xung đột: Cùng triệu chứng nhưng kết quả nào xuất hiện nhiều hơn thì chọn
        if frequency > rule_base[antecedent]['Tan_suat']:
            rule_base[antecedent] = {'Ket_luan': consequent, 'Tan_suat': frequency}

# ==========================================
# BƯỚC 4: RÚT GỌN LUẬT VÀ IN KẾT QUẢ
# ==========================================
# Chỉ lấy những luật xuất hiện từ 5 lần trở lên để loại bỏ nhiễu/ngoại lệ
MIN_SUPPORT = 5
final_rules = {k: v for k, v in rule_base.items() if v['Tan_suat'] >= MIN_SUPPORT}

print("\n=== BỘ LUẬT CHẨN ĐOÁN TIỂU ĐƯỜNG TỪ THUẬT TOÁN SQ-RuleExtract ===")
# Sắp xếp để in các luật "CÓ nguy cơ Tiểu đường" lên trước cho dễ chú ý
sorted_rules = sorted(final_rules.items(), key=lambda x: x[1]['Ket_luan'])

for idx, (antecedent, info) in enumerate(sorted_rules):
    conditions = [f"{features[i]} {antecedent[i]}" for i in range(len(antecedent))]
    if_part = " VÀ ".join(conditions)
    
    print(f"Luật {idx+1}:")
    print(f"   NẾU   [{if_part}]")
    print(f"   THÌ   Bệnh nhân [{info['Ket_luan']}]")
    print(f"   (Độ hỗ trợ: {info['Tan_suat']} hồ sơ bệnh án tương đồng)")
    print("-" * 70)