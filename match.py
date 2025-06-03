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

pic_path = "./pic/sheet6"
class CompanyNameMatcher:
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        
    def normalize_name(self, name):
        if not name:
            return ""
        
        name = str(name).strip()
        name = name.upper()
        name = unicodedata.normalize('NFKD', name)
        name = ''.join([c for c in name if not unicodedata.combining(c)])
        name = name.replace(' ', '')
        name = re.sub(r'[^A-Z0-9]', '', name)
        
        return name
    
    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def levenshtein_similarity(self, s1, s2):
        distance = self.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - (distance / max_len)
    
    def jaccard_similarity(self, s1, s2, n=2):
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
    
    def sequence_similarity(self, s1, s2):
        return SequenceMatcher(None, s1, s2).ratio()
    
    def calculate_similarity(self, name1, name2, use_normalized=True):
        if use_normalized:
            norm1 = self.normalize_name(name1)
            norm2 = self.normalize_name(name2)
            
            if norm1 == norm2:
                return 1.0
        else:
            norm1 = name1
            norm2 = name2
        
        lev_sim = self.levenshtein_similarity(norm1, norm2)
        jac_sim = self.jaccard_similarity(norm1, norm2)
        seq_sim = self.sequence_similarity(norm1, norm2)
        
        weights = [0.4, 0.3, 0.3]
        weighted_sim = (lev_sim * weights[0] + 
                       jac_sim * weights[1] + 
                       seq_sim * weights[2])
        
        return weighted_sim
    
    def find_matches(self, names_list, target_name=None):
        if target_name:
            matches = []
            for name in names_list:
                if name != target_name:
                    similarity = self.calculate_similarity(target_name, name)
                    if similarity >= self.similarity_threshold:
                        matches.append({
                            'name': name,
                            'similarity': similarity,
                            'normalized': self.normalize_name(name)
                        })
            
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            return matches
        
        else:
            matches = []
            processed = set()
            
            for i, name1 in enumerate(names_list):
                for j, name2 in enumerate(names_list[i+1:], i+1):
                    pair_key = tuple(sorted([i, j]))
                    if pair_key not in processed:
                        similarity = self.calculate_similarity(name1, name2)
                        if similarity >= self.similarity_threshold:
                            matches.append({
                                'name1': name1,
                                'name2': name2,
                                'similarity': similarity,
                                'normalized1': self.normalize_name(name1),
                                'normalized2': self.normalize_name(name2)
                            })
                        processed.add(pair_key)
            
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            return matches
    
    def deduplicate_names(self, names_list):
        groups = []
        processed = set()
        
        for i, name in enumerate(names_list):
            if i in processed:
                continue
            
            group = [name]
            processed.add(i)
            
            for j, other_name in enumerate(names_list[i+1:], i+1):
                if j not in processed:
                    similarity = self.calculate_similarity(name, other_name)
                    if similarity >= self.similarity_threshold:
                        group.append(other_name)
                        processed.add(j)
            
            groups.append({
                'representative': name,
                'members': group,
                'count': len(group)
            })
        
        return groups

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
                similarity1 = self.calculate_similarity(row2['NAME_normalized'], row1['NAME1_normalized'], use_normalized=False)
                similarity2 = self.calculate_similarity(row2['NAME_normalized'], row1['NAME2_normalized'], use_normalized=False)
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

        cm = confusion_matrix(all_labels, [1 if s >= self.similarity_threshold else 0 for s in all_similarities])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, pad=20)
        plt.savefig(pic_path + '/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

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

        thresholds = np.linspace(0, 1, 100)
        metrics_list = []
        for threshold in thresholds:
            preds = [1 if s >= threshold else 0 for s in all_similarities]
            tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            metrics_list.append({'threshold': threshold, 'precision': prec, 'recall': rec, 'f1': f1})
        metrics_df = pd.DataFrame(metrics_list)
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', color='#27AE60', lw=2)
        plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', color='#3498DB', lw=2)
        plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1 Score', color='#E74C3C', lw=2)
        plt.fill_between(metrics_df['threshold'], metrics_df['precision'], alpha=0.2, color='#27AE60')
        plt.fill_between(metrics_df['threshold'], metrics_df['recall'], alpha=0.2, color='#3498DB')
        plt.fill_between(metrics_df['threshold'], metrics_df['f1'], alpha=0.2, color='#E74C3C')
        plt.axvline(x=self.similarity_threshold, color='#7F8C8D', linestyle='--', 
                   label=f'Current Threshold ({self.similarity_threshold})', linewidth=2)
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
            mask = (np.array(all_similarities) >= lower) & (np.array(all_similarities) < upper)
            if np.any(mask):
                true_labels = np.array(all_labels)[mask]
                preds = [1 if s >= self.similarity_threshold else 0 for s in np.array(all_similarities)[mask]]
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
        print(f"{lower:.1f}-{upper:.1f}: TP={tp}, FP={fp}, FN={fn}")
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

        from sklearn.metrics import auc as sk_auc
        f1 = f1_score(all_labels, [1 if s >= self.similarity_threshold else 0 for s in all_similarities])
        accuracy = (np.array(all_labels) == np.array([1 if s >= self.similarity_threshold else 0 for s in all_similarities])).mean()
        precision_val = precision_score(all_labels, [1 if s >= self.similarity_threshold else 0 for s in all_similarities])
        recall_val = recall_score(all_labels, [1 if s >= self.similarity_threshold else 0 for s in all_similarities])
        roc_auc = sk_auc(fpr, tpr)
        pr_auc = sk_auc(recall, precision)
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision_val,
            'Recall': recall_val,
            'F1 Score': f1,
            'AUC-ROC': roc_auc,
            'AUC-PR': pr_auc
        }
        categories = list(metrics.keys())
        values = list(metrics.values())
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

        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            X = np.array(all_similarities).reshape(-1, 1)
            y = np.array(all_labels)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if X_scaled.shape[0] < 2 or X_scaled.shape[1] < 1:
                print("Too few samples for t-SNE/PCA visualization, skipping this plot.")
            else:
                if len(X_scaled) >= 3:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
                    X_vis = tsne.fit_transform(X_scaled)
                    method = 't-SNE'
                else:
                    pca = PCA(n_components=2)
                    X_vis = pca.fit_transform(X_scaled)
                    method = 'PCA'
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], 
                                    c=y, cmap='viridis', 
                                    alpha=0.6, 
                                    edgecolors='w',
                                    s=100)
                plt.colorbar(scatter, label='True Label')
                plt.title(f'{method} Visualization of Similarity Scores', fontsize=14, pad=20)
                plt.xlabel(f'{method} Component 1', fontsize=12)
                plt.ylabel(f'{method} Component 2', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.savefig(pic_path + '/tsne_visualization.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"t-SNE/PCA visualization could not be created: {str(e)}")
            print("This might be due to insufficient data points or memory constraints.")
        
        return metrics


def demo():
    matcher = CompanyNameMatcher(similarity_threshold=0.85)
    
    company_names = [
        "PADIERNA PENA, Luis Orlando", "PADIERNAPENALuisOrlando", "PADIERNA PEÑA Luis Orlando", "Padierna Pena Luis Orlando",
        "SMITH & JONES Corporation", "SMITH&JONES Corporation", "Smith Jones Corp",
        "ABC Company Limited", "ABC Company Ltd.", "ABC COMPANY LIMITED", "A.B.C. CO LTD", "A B C COMPANY LIMITED",
        "Tesla Inc", "TESLA Incorporated", "Tesla Motors", "TESLA", "Telsa", "Telsa Inc",
        "Google LLC", "GOOGLE", "Google Inc", "Alphabet Inc", "ALPHABET", "Alphabet",
        "Microsoft Corp", "MICROSOFT CORPORATION", "MSFT", "Microsoft", "Micro Soft", "MICROSOFT CO"
    ]
    labels = [1]*7 + [2]*5 + [3]*6 + [4]*6 + [5]*6
    
    similarities = []
    for i in range(len(company_names)):
        for j in range(i+1, len(company_names)):
            sim = matcher.calculate_similarity(company_names[i], company_names[j])
            similarities.append(sim)
    
    label_pairs = []
    for i in range(len(company_names)):
        for j in range(i+1, len(company_names)):
            label_pairs.append(1 if labels[i] == labels[j] else 0)
    
    pic_path = './demo_pic'
    os.makedirs(pic_path, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    fpr, tpr, thresholds = roc_curve(label_pairs, similarities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='#2E86C1', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86C1')
    plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (Demo)', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{pic_path}/demo_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    precision, recall, thresholds = precision_recall_curve(label_pairs, similarities)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(10, 6))
    if len(recall) > 2:
        plt.step(recall, precision, where='post', color='#27AE60', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='#27AE60')
    else:
        plt.plot(recall, precision, color='#27AE60', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Demo)', fontsize=14, pad=20)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{pic_path}/demo_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    predictions = [1 if s >= matcher.similarity_threshold else 0 for s in similarities]
    cm = confusion_matrix(label_pairs, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix (Demo)', fontsize=14, pad=20)
    plt.savefig(f'{pic_path}/demo_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.75, color='#3498DB', edgecolor='white')
    plt.axvline(x=matcher.similarity_threshold, color='#E74C3C', linestyle='--', 
               label=f'Threshold ({matcher.similarity_threshold})', linewidth=2)
    plt.text(matcher.similarity_threshold+0.01, plt.ylim()[1]*0.9, f'Threshold\n({matcher.similarity_threshold})',
             color='#E74C3C', fontsize=11, rotation=90, va='top', ha='left', fontweight='bold', alpha=0.8)
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Similarity Scores (Demo)', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{pic_path}/demo_similarity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    thresholds = np.linspace(0, 1, 100)
    metrics_list = []
    for threshold in thresholds:
        preds = [1 if s >= threshold else 0 for s in similarities]
        tn, fp, fn, tp = confusion_matrix(label_pairs, preds).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        metrics_list.append({'threshold': threshold, 'precision': prec, 'recall': rec, 'f1': f1})
    metrics_df = pd.DataFrame(metrics_list)
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', color='#27AE60', lw=2)
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', color='#3498DB', lw=2)
    plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1 Score', color='#E74C3C', lw=2)
    plt.fill_between(metrics_df['threshold'], metrics_df['precision'], alpha=0.2, color='#27AE60')
    plt.fill_between(metrics_df['threshold'], metrics_df['recall'], alpha=0.2, color='#3498DB')
    plt.fill_between(metrics_df['threshold'], metrics_df['f1'], alpha=0.2, color='#E74C3C')
    plt.axvline(x=matcher.similarity_threshold, color='#7F8C8D', linestyle='--', 
               label=f'Current Threshold ({matcher.similarity_threshold})', linewidth=2)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics vs Threshold (Demo)', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{pic_path}/demo_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    similarity_ranges = np.linspace(0, 1, 11)
    error_analysis = []
    for i in range(len(similarity_ranges)-1):
        lower, upper = similarity_ranges[i], similarity_ranges[i+1]
        mask = (np.array(similarities) >= lower) & (np.array(similarities) < upper)
        if np.any(mask):
            true_labels = np.array(label_pairs)[mask]
            preds = [1 if s >= matcher.similarity_threshold else 0 for s in np.array(similarities)[mask]]
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
    plt.title('Error Distribution by Similarity Score Range (Demo)', fontsize=14, pad=20)
    plt.xticks(x, error_df['range'], rotation=45)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{pic_path}/demo_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    from sklearn.metrics import auc as sk_auc
    f1 = f1_score(label_pairs, [1 if s >= matcher.similarity_threshold else 0 for s in similarities])
    accuracy = (np.array(label_pairs) == np.array([1 if s >= matcher.similarity_threshold else 0 for s in similarities])).mean()
    precision_val = precision_score(label_pairs, [1 if s >= matcher.similarity_threshold else 0 for s in similarities])
    recall_val = recall_score(label_pairs, [1 if s >= matcher.similarity_threshold else 0 for s in similarities])
    roc_auc = sk_auc(fpr, tpr)
    pr_auc = sk_auc(recall, precision)
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision_val,
        'Recall': recall_val,
        'F1 Score': f1,
        'AUC-ROC': roc_auc,
        'AUC-PR': pr_auc
    }
    categories = list(metrics.keys())
    values = list(metrics.values())
    values += [values[0]]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86C1')
    ax.fill(angles, values, alpha=0.25, color='#2E86C1')
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=12)
    ax.set_ylim(0, 1)
    plt.title('Performance Metrics Radar Chart (Demo)', fontsize=14, pad=20)
    plt.savefig(f'{pic_path}/demo_performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = np.array(similarities).reshape(-1, 1)
        y = np.array(label_pairs)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if X_scaled.shape[0] < 3 or X_scaled.shape[1] < 2:
            print("Too few samples/features for t-SNE/PCA visualization, skipping this plot.")
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X_scaled)-1))
            X_vis = tsne.fit_transform(X_scaled)
            method = 't-SNE'
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], 
                                c=y, cmap='viridis', 
                                alpha=0.6, 
                                edgecolors='w',
                                s=100)
            plt.colorbar(scatter, label='True Label')
            plt.title(f'{method} Visualization (Demo)', fontsize=14, pad=20)
            plt.xlabel(f'{method} Component 1', fontsize=12)
            plt.ylabel(f'{method} Component 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{pic_path}/demo_tsne_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"t-SNE/PCA visualization could not be created: {str(e)}")

    print(f"Demo visualizations have been saved in {pic_path}. Please check the generated PNG files.")


if __name__ == "__main__":
    matcher = CompanyNameMatcher(similarity_threshold=0.85)
    
    metrics = matcher.link_datasets(
        df1_path="main.csv",
        df2_path="./csv_output/Sheet6_filtered.csv",
        output_path="linked_results-06.csv"
    )