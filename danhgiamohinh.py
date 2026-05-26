import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score

def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    
    features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    target = 'Outcome'

    for col in ['Glucose', 'BloodPressure', 'BMI']:
        df = df[df[col] > 0]
        
    return df[features + [target]].copy(), features, target

def train_sq_rule_extract(df_train, features, target, min_support=5):
    """Giai đoạn học tham số Đại số gia tử và trích xuất luật trên tập Train"""
    ha_params = {}
    df_mapped = df_train.copy()
    
    for col in features:
        col_min, col_max = df_train[col].min(), df_train[col].max()
        # Chuẩn hóa nội bộ tập Train về [0, 1]
        x_norm = (df_train[col] - col_min) / (col_max - col_min)
        
        m = x_norm.mean()
        fm_c_minus = len(x_norm[x_norm <= m]) / len(x_norm)
        fm_c_plus = 1.0 - fm_c_minus
        
        m_low = x_norm[x_norm <= m].mean()
        m_high = x_norm[x_norm > m].mean()
        p_low = len(x_norm[x_norm <= m_low]) / len(x_norm[x_norm <= m])
        p_high = len(x_norm[x_norm >= m_high]) / len(x_norm[x_norm > m])
        mu_rat = (p_low + p_high) / 2.0
        mu_hoi = 1.0 - mu_rat
        
        b1 = mu_rat * fm_c_minus
        b2 = fm_c_minus
        b3 = fm_c_minus + mu_hoi * fm_c_plus
        
        ha_params[col] = {'min': col_min, 'max': col_max, 'b1': b1, 'b2': b2, 'b3': b3}
        
        def to_linguistic(val):
            v = (val - col_min) / (col_max - col_min)
            if v <= b1: return "Thấp"
            elif v <= b3: return "Bình thường"
            else: return "Cao"
        
        df_mapped[col + '_Lang'] = df_train[col].apply(to_linguistic)
        
    ling_cols = [col + '_Lang' for col in features]
    rules_grouped = df_mapped.groupby(ling_cols + [target]).size().reset_index(name='Count')
    
    raw_rules = {}
    for _, row in rules_grouped.iterrows():
        antecedent = tuple(row[ling_cols])
        consequent = row[target]
        count = row['Count']
        if antecedent not in raw_rules or count > raw_rules[antecedent]['Count']:
            raw_rules[antecedent] = {'predict': consequent, 'Count': count}
            
    final_rules = {k: v['predict'] for k, v in raw_rules.items() if v['Count'] >= min_support}
    return final_rules, ha_params

def predict_sq_rule_extract(X_test, features, final_rules, ha_params, default_class=0):
    """Áp dụng bộ luật đã học để dự đoán trên tập dữ liệu kiểm thử hoàn toàn mới"""
    y_pred = []
    
    for _, row in X_test.iterrows():
        # Chuyển đổi dữ liệu số của ca kiểm thử sang nhãn ngữ nghĩa dựa trên bẫy HA đã học
        test_antecedent = []
        for col in features:
            val = row[col]
            p = ha_params[col]
            v = (val - p['min']) / (p['max'] - p['min']) if (p['max'] - p['min']) > 0 else 0
            
            if v <= p['b1']: test_antecedent.append("Thấp")
            elif v <= p['b3']: test_antecedent.append("Bình thường")
            else: test_antecedent.append("Cao")
            
        test_tuple = tuple(test_antecedent)
        
        # Suy diễn luật: Nếu khớp luật thì lấy kết luận của luật, nếu không khớp thì dùng nhãn mặc định
        if test_tuple in final_rules:
            y_pred.append(final_rules[test_tuple])
        else:
            y_pred.append(default_class)
            
    return np.array(y_pred)

if __name__ == "__main__":
    df, features, target = load_data()
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    df_train_combined = pd.concat([X_train, y_train], axis=1)
    rules, ha_parameters = train_sq_rule_extract(df_train_combined, features, target)
    y_pred_sq = predict_sq_rule_extract(X_test, features, rules, ha_parameters)
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    mlp_model = MLPClassifier(max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_test)
    
    models_metrics = {
        "Mạng Nơ-ron": (y_test, y_pred_mlp),
        "Random Forest": (y_test, y_pred_rf),
        "SQ-RuleExtract": (y_test, y_pred_sq)
    }
    
    for name, (gt, pred) in models_metrics.items():
        acc = accuracy_score(gt, pred) * 100
        rec = recall_score(gt, pred) * 100
        print(f"Mô hình: {name}")
        print(f"  -> Độ chính xác (Accuracy): {acc:.1f}%")
        print(f"  -> Độ nhạy (Recall):        {rec:.1f}%")
        print("-" * 40)