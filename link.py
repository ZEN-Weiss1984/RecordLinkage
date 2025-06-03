import linktransformer as lt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score, precision_score, recall_score

pic_path = "./pic/sheet7"
os.makedirs(pic_path, exist_ok=True)
if not os.path.exists(pic_path):
    raise Exception(f"Failed to create directory: {pic_path}")

similarity_threshold = 0.8

df1 = pd.read_csv("main.csv")
df2 = pd.read_csv("csv_output/Sheet7_filtered.csv")

print("df1的列名:", df1.columns.tolist())
print("\ndf2的列名:", df2.columns.tolist())

print("\ndf1的前几行:")
print(df1.head())
print("\ndf2的前几行:")
print(df2.head())

print("正在进行数据链接...")

def clean_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = ''.join(e for e in name if e.isalnum() or e.isspace() or e in '.,-')
    name = ' '.join(name.split())
    return name.strip()

def get_name_variations(name):
    if pd.isna(name):
        return []
    name = clean_name(name)
    variations = [name]
    
    no_punct = ''.join(e for e in name if e.isalnum() or e.isspace())
    if no_punct != name:
        variations.append(no_punct)
    
    no_space = ''.join(e for e in name if e.isalnum() or e in '.,-')
    if no_space != name:
        variations.append(no_space)
    
    if ',' in name:
        parts = name.split(',')
        if len(parts) == 2:
            variations.append(parts[1].strip() + ' ' + parts[0].strip())
    
    return variations

df1['NAME1_clean'] = df1['NAME1'].apply(clean_name)
df1['NAME2_clean'] = df1['NAME2'].apply(clean_name)
df2['NAME_clean'] = df2['NAME'].apply(clean_name)

df1_name1 = df1[['ID', 'NAME1', 'NAME1_clean']].copy()
df1_name1.rename(columns={'NAME1': 'NAME', 'NAME1_clean': 'NAME_clean'}, inplace=True)

df1_name2 = df1[['ID', 'NAME2', 'NAME2_clean']].copy()
df1_name2.rename(columns={'NAME2': 'NAME', 'NAME2_clean': 'NAME_clean'}, inplace=True)

print("正在匹配NAME1...")
df_matched_name1 = lt.merge(df2, df1_name1, merge_type='1:m', on="NAME", 
                           model="sentence-transformers/all-MiniLM-L6-v2")

print("正在匹配NAME2...")
df_matched_name2 = lt.merge(df2, df1_name2, merge_type='1:m', on="NAME", 
                           model="sentence-transformers/all-MiniLM-L6-v2")

df_matched_combined = pd.concat([df_matched_name1, df_matched_name2], ignore_index=True)

similarity_col = [col for col in df_matched_combined.columns if 'score' in col.lower() or 'similarity' in col.lower()][0]
print(f"\n使用相似度列: {similarity_col}")

df_matched_combined['exact_match'] = df_matched_combined.apply(
    lambda row: clean_name(row['NAME_x']) == clean_name(row['NAME_y']), axis=1
)

df_matched_combined['name_variations_match'] = df_matched_combined.apply(
    lambda row: any(
        var in get_name_variations(row['NAME_x']) 
        for var in get_name_variations(row['NAME_y'])
    ), axis=1
)

df_matched_combined = df_matched_combined[
    (df_matched_combined[similarity_col] >= similarity_threshold) |
    (df_matched_combined['exact_match'] == True) |
    (df_matched_combined['name_variations_match'] == True)
]

all_similarities = df_matched_combined[similarity_col].values
all_labels = (df_matched_combined['ID_x'] == df_matched_combined['ID_y']).astype(int).values

df_matched_combined = df_matched_combined.sort_values(similarity_col, ascending=False)
df_matched_combined = df_matched_combined.drop_duplicates(subset=['ID_x'], keep='first')

df_matched_final = pd.merge(df_matched_combined, df1, left_on='ID_y', right_on='ID', how='left')

df_matched_final.to_csv("linked_results-07.csv", index=False)

print("\n合并后的数据框列名:", df_matched_final.columns.tolist())
print("\n合并后的数据框前几行:")
print(df_matched_final.head())

print("计算评估指标...")

df_matched_final['ID_x'] = df_matched_final['ID_x'].astype(int)
df_matched_final['ID_y'] = df_matched_final['ID_y'].astype(int)

df2_ids = df2['ID'].values
df1_ids = df1['ID'].values

id_pairs = pd.DataFrame({
    'ID_x': np.repeat(df2_ids, len(df1_ids)),
    'ID_y': np.tile(df1_ids, len(df2_ids))
})

print("从匹配结果中获取相似度分数...")
all_pairs_similarities = []
all_pairs_labels = []

df_all_matches = pd.concat([df_matched_name1, df_matched_name2], ignore_index=True)

for _, pair in tqdm(id_pairs.iterrows(), total=len(id_pairs), desc="处理数据对"):
    match = df_all_matches[
        (df_all_matches['ID_x'] == pair['ID_x']) & 
        (df_all_matches['ID_y'] == pair['ID_y'])
    ]
    
    if len(match) > 0:
        similarity = match['score'].max()
    else:
        similarity = 0.0
    
    all_pairs_similarities.append(similarity)
    all_pairs_labels.append(1 if pair['ID_x'] == pair['ID_y'] else 0)

all_pairs_similarities = np.array(all_pairs_similarities)
all_pairs_labels = np.array(all_pairs_labels)

id_pairs['is_linked'] = False
for _, row in tqdm(df_matched_final.iterrows(), total=len(df_matched_final), desc="标记链接对"):
    mask = (id_pairs['ID_x'] == row['ID_x']) & (id_pairs['ID_y'] == row['ID_y'])
    id_pairs.loc[mask, 'is_linked'] = True

id_pairs['is_same_id'] = id_pairs['ID_x'] == id_pairs['ID_y']

true_positives = ((id_pairs['is_linked']) & (id_pairs['is_same_id'])).sum()
false_positives = ((id_pairs['is_linked']) & (~id_pairs['is_same_id'])).sum()
false_negatives = ((~id_pairs['is_linked']) & (id_pairs['is_same_id'])).sum()
true_negatives = ((~id_pairs['is_linked']) & (~id_pairs['is_same_id'])).sum()

total = true_positives + false_positives + false_negatives + true_negatives
accuracy = float((true_positives + true_negatives) / total)
precision = float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0.0
recall = float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0.0
f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

print("\n评估指标:")
print(f"True Positives (正确链接的匹配): {true_positives}")
print(f"False Positives (错误链接的不匹配): {false_positives}")
print(f"False Negatives (未链接的匹配): {false_negatives}")
print(f"True Negatives (正确未链接的不匹配): {true_negatives}")
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

plt.style.use('seaborn-v0_8-whitegrid')

fpr, tpr, thresholds = roc_curve(all_pairs_labels, all_pairs_similarities)
roc_auc = float(auc(fpr, tpr))
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='#2E86C1', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86C1')
plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(pic_path + '/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

precision_curve, recall_curve, thresholds = precision_recall_curve(all_pairs_labels, all_pairs_similarities)
pr_auc = float(auc(recall_curve, precision_curve))
plt.figure(figsize=(10, 6))
if len(recall_curve) > 2:
    plt.step(recall_curve, precision_curve, where='post', color='#27AE60', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.fill_between(recall_curve, precision_curve, step='post', alpha=0.2, color='#27AE60')
else:
    plt.plot(recall_curve, precision_curve, color='#27AE60', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, pad=20)
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(pic_path + '/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()

cm = confusion_matrix(all_pairs_labels, [1 if s >= similarity_threshold else 0 for s in all_pairs_similarities])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Confusion Matrix', fontsize=14, pad=20)
plt.savefig(pic_path + '/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'AUC-ROC': roc_auc,
    'AUC-PR': pr_auc
}
categories = list(metrics.keys())
values = list(metrics.values())
print(f"Radar chart categories order: {categories}")
print(f"Radar chart values order: {values}")
values += [values[0]]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, values, 'o-', linewidth=2, color='#2E86C1')
ax.fill(angles, values, alpha=0.25, color='#2E86C1')
ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=12)
ax.set_ylim(0, 1)
plt.title('Performance Metrics Radar Chart', fontsize=14, pad=20)
plt.savefig(pic_path + '/performance_radar.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(all_pairs_similarities, bins=50, alpha=0.75, color='#3498DB', edgecolor='white')
plt.axvline(x=similarity_threshold, color='#E74C3C', linestyle='--', 
           label=f'Threshold ({similarity_threshold})', linewidth=2)
plt.text(similarity_threshold+0.01, plt.ylim()[1]*0.9, f'Threshold\n({similarity_threshold})',
         color='#E74C3C', fontsize=11, rotation=90, va='top', ha='left', fontweight='bold', alpha=0.8)
plt.xlabel('Similarity Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Similarity Scores', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(pic_path + '/similarity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

thresholds = np.linspace(0, 1, 100)
metrics_list = []
for threshold in thresholds:
    preds = [1 if s >= threshold else 0 for s in all_pairs_similarities]
    tn, fp, fn, tp = confusion_matrix(all_pairs_labels, preds).ravel()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
    metrics_list.append({'threshold': threshold, 'precision': prec, 'recall': rec, 'f1': f1})
metrics_df = pd.DataFrame(metrics_list)
plt.figure(figsize=(12, 6))
plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', color='#27AE60', lw=2)
plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', color='#3498DB', lw=2)
plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1 Score', color='#E74C3C', lw=2)
plt.fill_between(metrics_df['threshold'], metrics_df['precision'], alpha=0.2, color='#27AE60')
plt.fill_between(metrics_df['threshold'], metrics_df['recall'], alpha=0.2, color='#3498DB')
plt.fill_between(metrics_df['threshold'], metrics_df['f1'], alpha=0.2, color='#E74C3C')
plt.axvline(x=similarity_threshold, color='#7F8C8D', linestyle='--', 
           label=f'Current Threshold ({similarity_threshold})', linewidth=2)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Performance Metrics vs Threshold', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(pic_path + '/threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

similarity_ranges = np.linspace(0, 1, 11)
error_analysis = []
for i in range(len(similarity_ranges)-1):
    lower, upper = similarity_ranges[i], similarity_ranges[i+1]
    mask = (np.array(df_matched_final[similarity_col]) >= lower) & (np.array(df_matched_final[similarity_col]) < upper)
    if np.any(mask):
        true_labels = np.array(df_matched_final['exact_match'])[mask]
        preds = [1 if s >= similarity_threshold else 0 for s in np.array(df_matched_final[similarity_col])[mask]]
        cm = confusion_matrix(true_labels, preds, labels=[0,1])
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        error_analysis.append({
            'range': f'{lower:.1f}-{upper:.1f}',
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        })
error_df = pd.DataFrame(error_analysis)
plt.figure(figsize=(12, 6))
x = np.arange(len(error_df))
width = 0.25
plt.bar(x - width, error_df['true_positives'], width, label='True Positives', color='#27AE60', alpha=0.8)
plt.bar(x, error_df['false_positives'], width, label='False Positives', color='#E74C3C', alpha=0.8)
plt.bar(x + width, error_df['false_negatives'], width, label='False Negatives', color='#F1C40F', alpha=0.8)
plt.xlabel('Similarity Score Range', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Error Distribution by Similarity Score Range', fontsize=14, pad=20)
plt.xticks(x, error_df['range'], rotation=45)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(pic_path + '/error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n可视化结果已保存到 {pic_path} 目录")


