import recordlinkage
# from recordlinkage.recordlinkage.datasets import load_febrl1
import pandas as pd
from tqdm import tqdm  
import json
filepath = "./recordlinkage/primary.csv"

def load_data():
    data = pd.read_csv(filepath, dtype=str)
    # data.set_index("NAME", inplace=True)
    print(f"[DEBUG] 数据加载完成，共 {len(data)} 条记录")
    return data

dfA = load_data()

print("[DEBUG] 开始索引生成...")
indexer = recordlinkage.Index()
indexer.full()
candidate_links = indexer.index(dfA)
# indexer.block(left_on="TYPE")
print(f"[DEBUG] 索引生成完成，候选对数量: {len(candidate_links)}")

print("[DEBUG] 开始特征比对...")
compare_cl = recordlinkage.Compare()
compare_cl.string("NAME", "NAME", method="jarowinkler", threshold=0.95, label="NAME")
# compare_cl.exact("NAME", "NAME", label="NAME")
# compare_cl.exact("ID", "ID", label="ID")
# compare_cl.exact("TYPE", "TYPE", label="TYPE")



features = compare_cl.compute(candidate_links, dfA)
print(f"[DEBUG] 特征比对完成，特征矩阵形状: {features.shape}")

matches = features[(features["NAME"] == 1)]
print(f"[DEBUG] 检测到重复项对数: {len(matches)}")

for i, idx_pair in enumerate(matches.index, 1):
    idx1, idx2 = idx_pair
    name1 = dfA.loc[idx1, "NAME"]
    # id1 = dfA.loc[idx1, "ID"]
    # type1 = dfA.loc[idx1, "TYPE"]
    name2 = dfA.loc[idx2, "NAME"]
    # type2 = dfA.loc[idx2, "TYPE"]
    # id2 = dfA.loc[idx2, "ID"]
    # print(f"[DEBUG] Duplicate pair {i}: NAME1={name1}, TYPE1={type1} | NAME2={name2}, TYPE2={type2}")
    print(f"[DEBUG] Duplicate pair {i}: NAME1={name1} | NAME2={name2}")



duplicate_indices = set()
duplicate_info = {} 

for i, idx_pair in enumerate(matches.index, 1):
    idx1, idx2 = idx_pair
    duplicate_indices.add(idx2)  
    
    duplicate_info[f"pair_{i}"] = {
        "idx1": str(idx1),
        "name1": dfA.loc[idx1, "NAME"],
        "id1": dfA.loc[idx1, "ID"],
        "idx2": str(idx2),
        "name2": dfA.loc[idx2, "NAME"],
        "id2": dfA.loc[idx2, "ID"]
    }

with open("./recordlinkage/duplicate_indices.json", "w") as f:
    json.dump(duplicate_info, f, ensure_ascii=False, indent=4)  



df_deduplicated = dfA.drop(index=list(duplicate_indices))


print(f"[DEBUG] 去重后剩余记录数: {len(df_deduplicated)}")

output_file = "./recordlinkage/deduplicated_primary.csv"
df_deduplicated.to_csv(output_file, index=False)
print(f"[DEBUG] 去重后的数据已保存到: {output_file}")