import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score, precision_score, recall_score
import os

pic_path = "./pic/sheet—02"

class ShuffledNameMatcher:
    
    def __init__(self, similarity_threshold=0.85):

        self.similarity_threshold = similarity_threshold
        
    def normalize_name(self, name):
        if not name:
            return ""
        
        # 转换为字符串并去除首尾空白
        name = str(name).strip()
        
        # 转换为大写
        name = name.upper()
        
        # 规范化Unicode字符（处理重音符号等）
        name = unicodedata.normalize('NFKD', name)
        name = ''.join([c for c in name if not unicodedata.combining(c)])
        
        # 移除所有非字母数字字符和空格
        name = re.sub(r'[^A-Z0-9]', '', name)
        
        return name
    
    def get_character_frequency(self, text):
        return Counter(text)
    
    def character_frequency_similarity(self, s1, s2):

        freq1 = self.get_character_frequency(s1)
        freq2 = self.get_character_frequency(s2)
        
        
        all_chars = set(freq1.keys()) | set(freq2.keys())
        dot_product = sum(freq1.get(c, 0) * freq2.get(c, 0) for c in all_chars)
        norm1 = sum(freq1.get(c, 0) ** 2 for c in all_chars) ** 0.5
        norm2 = sum(freq2.get(c, 0) ** 2 for c in all_chars) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def ngram_similarity(self, s1, s2, n=2):
        def get_ngrams(s, n):
            return set([s[i:i+n] for i in range(len(s)-n+1)])
        
        if len(s1) < n or len(s2) < n:
            return 1.0 if s1 == s2 else 0.0
        
        ngrams1 = get_ngrams(s1, n)
        ngrams2 = get_ngrams(s2, n)
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        if len(union) == 0:
            return 0.0
        return len(intersection) / len(union)
    
    def calculate_similarity(self, name1, name2):
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)

        if norm1 == norm2:
            return 1.0
        
        char_freq_sim = self.character_frequency_similarity(norm1, norm2)
        ngram_sim = self.ngram_similarity(norm1, norm2)
        
        weights = [0.7, 0.3] 
        weighted_sim = (char_freq_sim * weights[0] + 
                       ngram_sim * weights[1])
        
        return weighted_sim
    
    def link_datasets(self, df1_path, df2_path, output_path="linked_results.csv"):
        df1 = pd.read_csv(df1_path)
        df2 = pd.read_csv(df2_path)
        
        print("df1的列名:", df1.columns.tolist())
        print("\ndf2的列名:", df2.columns.tolist())
        print("\ndf1的前几行:")
        print(df1.head())
        print("\ndf2的前几行:")
        print(df2.head())
        
        print("\n正在进行名称标准化...")
        df1['NAME1_normalized'] = df1['NAME1'].apply(self.normalize_name)
        df1['NAME2_normalized'] = df1['NAME2'].apply(self.normalize_name)
        df2['NAME_normalized'] = df2['NAME'].apply(self.normalize_name)
        
        print("\n标准化后的df1前几行:")
        print(df1[['NAME1', 'NAME1_normalized', 'NAME2', 'NAME2_normalized']].head())
        print("\n标准化后的df2前几行:")
        print(df2[['NAME', 'NAME_normalized']].head())
        
        print("\n正在进行数据链接...")
        
        linked_results = []
        all_similarities = []
        all_labels = []
        
        for _, row2 in tqdm(df2.iterrows(), total=len(df2), desc="链接数据"):
            best_match = None
            best_similarity = 0
            
            for _, row1 in df1.iterrows():
                similarity1 = self.calculate_similarity(row2['NAME_normalized'], row1['NAME1_normalized'])
                similarity2 = self.calculate_similarity(row2['NAME_normalized'], row1['NAME2_normalized'])
                similarity = max(similarity1, similarity2)
                
                all_similarities.append(similarity)
                all_labels.append(1 if row2['ID'] == row1['ID'] else 0)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = row1
            
            if best_match is not None:
                result_row = {}
                for col in df2.columns:
                    result_row[f"{col}_x"] = row2[col]
                for col in df1.columns:
                    result_row[f"{col}_y"] = best_match[col]
                result_row['similarity_score'] = best_similarity
                linked_results.append(result_row)
        
        df_linked = pd.DataFrame(linked_results)
        
        df_linked.to_csv(output_path, index=False)
        
        df_linked['ID_x'] = df_linked['ID_x'].astype(int)
        df_linked['ID_y'] = df_linked['ID_y'].astype(int)
        
        print("计算评估指标...")
        
        df2_ids = df2['ID'].values
        df1_ids = df1['ID'].values
        
        id_pairs = pd.DataFrame({
            'ID_x': np.repeat(df2_ids, len(df1_ids)),
            'ID_y': np.tile(df1_ids, len(df2_ids))
        })
        
        id_pairs['is_linked'] = False
        for _, row in tqdm(df_linked.iterrows(), total=len(df_linked), desc="标记链接对"):
            mask = (id_pairs['ID_x'] == row['ID_x']) & (id_pairs['ID_y'] == row['ID_y'])
            id_pairs.loc[mask, 'is_linked'] = True
        
        id_pairs['is_same_id'] = id_pairs['ID_x'] == id_pairs['ID_y']
        
        true_positives = ((id_pairs['is_linked']) & (id_pairs['is_same_id'])).sum()
        false_positives = ((id_pairs['is_linked']) & (~id_pairs['is_same_id'])).sum()
        false_negatives = ((~id_pairs['is_linked']) & (id_pairs['is_same_id'])).sum()
        true_negatives = ((~id_pairs['is_linked']) & (~id_pairs['is_same_id'])).sum()
        
        total = true_positives + false_positives + false_negatives + true_negatives
        accuracy = (true_positives + true_negatives) / total
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = f1_score(all_labels, [1 if s >= self.similarity_threshold else 0 for s in all_similarities])
        
        metrics = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
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
        os.makedirs(pic_path, exist_ok=True)

        # 1. ROC
        fpr, tpr, thresholds = roc_curve(all_labels, all_similarities)
        roc_auc = auc(fpr, tpr)
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

        # 2. Precision-Recall
        precision, recall, thresholds = precision_recall_curve(all_labels, all_similarities)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(10, 6))
        if len(recall) > 2:
            plt.step(recall, precision, where='post', color='#27AE60', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='#27AE60')
        else:
            plt.plot(recall, precision, color='#27AE60', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, pad=20)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(pic_path + '/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. confusion_matrix
        cm = confusion_matrix(all_labels, [1 if s >= self.similarity_threshold else 0 for s in all_similarities])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, pad=20)
        plt.savefig(pic_path + '/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 相似度分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(all_similarities, bins=50, alpha=0.75, color='#3498DB', edgecolor='white')
        plt.axvline(x=self.similarity_threshold, color='#E74C3C', linestyle='--', 
                   label=f'Threshold ({self.similarity_threshold})', linewidth=2)
        plt.text(self.similarity_threshold+0.01, plt.ylim()[1]*0.9, f'Threshold\n({self.similarity_threshold})',
                 color='#E74C3C', fontsize=11, rotation=90, va='top', ha='left', fontweight='bold', alpha=0.8)
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Similarity Scores', fontsize=14, pad=20)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(pic_path + '/similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        return metrics


if __name__ == "__main__":
    matcher = ShuffledNameMatcher(similarity_threshold=0.85)
    
    metrics = matcher.link_datasets(
        df1_path="main.csv",
        df2_path="./csv_output/Sheet5_filtered.csv",
        output_path="linked_results-05-02.csv"
    ) 