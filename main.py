import pandas as pd

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df_full = pd.read_csv(url)

features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
target = 'Outcome'
df = df_full[features + [target]].copy()


for col in features:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

def linguistic_mapping(val):
    if 0 <= val <= 0.35:
        return "Thấp"
    elif 0.35 < val <= 0.65:
        return "Bình thường" 
    else:
        return "Cao"

ling_cols = []
for col in features:
    ling_col_name = col + '_NgonNgu'
    df[ling_col_name] = df[col].apply(linguistic_mapping)
    ling_cols.append(ling_col_name)

print("Đang phân hoạch ngữ nghĩa và trích xuất luật...")
rules_df = df.groupby(ling_cols + [target]).size().reset_index(name='Tan_suat')

rule_base = {}
for index, row in rules_df.iterrows():
    antecedent = tuple(row[ling_cols])
    consequent = "CÓ nguy cơ Tiểu đường" if row[target] == 1 else "KHÔNG bị Tiểu đường"
    frequency = row['Tan_suat']
    
    if antecedent not in rule_base:
        rule_base[antecedent] = {'Ket_luan': consequent, 'Tan_suat': frequency}
    else:
        if frequency > rule_base[antecedent]['Tan_suat']:
            rule_base[antecedent] = {'Ket_luan': consequent, 'Tan_suat': frequency}


MIN_SUPPORT = 5
final_rules = {k: v for k, v in rule_base.items() if v['Tan_suat'] >= MIN_SUPPORT}

print("\n=== BỘ LUẬT CHẨN ĐOÁN TIỂU ĐƯỜNG TỪ THUẬT TOÁN SQ-RuleExtract ===")
sorted_rules = sorted(final_rules.items(), key=lambda x: x[1]['Ket_luan'])

for idx, (antecedent, info) in enumerate(sorted_rules):
    conditions = [f"{features[i]} {antecedent[i]}" for i in range(len(antecedent))]
    if_part = " VÀ ".join(conditions)
    
    print(f"Luật {idx+1}:")
    print(f"   NẾU   [{if_part}]")
    print(f"   THÌ   Bệnh nhân [{info['Ket_luan']}]")
    print(f"   (Độ hỗ trợ: {info['Tan_suat']} hồ sơ bệnh án tương đồng)")
    print("-" * 70)