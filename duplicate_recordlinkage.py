import recordlinkage
# from recordlinkage.recordlinkage.datasets import load_febrl1
import pandas as pd
from tqdm import tqdm  
import json
import argparse
import logging
import os

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data(filepath):
    data = pd.read_csv(filepath, dtype=str)
    # data.set_index("NAME", inplace=True)
    logger.debug(f"数据加载完成，共 {len(data)} 条记录")
    return data

def main():
    parser = argparse.ArgumentParser(description='Record linkage based deduplication')
    parser.add_argument('--in', dest='input_file', required=True, help='Input CSV file path')
    parser.add_argument('--out', dest='output_file', required=True, help='Output CSV file path')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    logger = setup_logging()
    
    dfA = load_data(args.input_file)

    logger.debug("开始索引生成...")
    indexer = recordlinkage.Index()
    indexer.full()
    candidate_links = indexer.index(dfA)
    # indexer.block(left_on="TYPE")
    logger.debug(f"索引生成完成，候选对数量: {len(candidate_links)}")

    logger.debug("开始特征比对...")
    compare_cl = recordlinkage.Compare()
    compare_cl.string("NAME", "NAME", method="jarowinkler", threshold=0.95, label="NAME")
    # compare_cl.exact("NAME", "NAME", label="NAME")
    # compare_cl.exact("ID", "ID", label="ID")
    # compare_cl.exact("TYPE", "TYPE", label="TYPE")

    features = compare_cl.compute(candidate_links, dfA)
    logger.debug(f"特征比对完成，特征矩阵形状: {features.shape}")

    matches = features[(features["NAME"] == 1)]
    logger.debug(f"检测到重复项对数: {len(matches)}")

    for i, idx_pair in enumerate(matches.index, 1):
        idx1, idx2 = idx_pair
        name1 = dfA.loc[idx1, "NAME"]
        # id1 = dfA.loc[idx1, "ID"]
        # type1 = dfA.loc[idx1, "TYPE"]
        name2 = dfA.loc[idx2, "NAME"]
        # type2 = dfA.loc[idx2, "TYPE"]
        # id2 = dfA.loc[idx2, "ID"]
        # print(f"[DEBUG] Duplicate pair {i}: NAME1={name1}, TYPE1={type1} | NAME2={name2}, TYPE2={type2}")
        logger.debug(f"Duplicate pair {i}: NAME1={name1} | NAME2={name2}")

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

    # Save duplicate info to JSON
    json_output = os.path.join(os.path.dirname(args.output_file), 'duplicate_indices.json')
    with open(json_output, "w") as f:
        json.dump(duplicate_info, f, ensure_ascii=False, indent=4)  

    df_deduplicated = dfA.drop(index=list(duplicate_indices))

    logger.debug(f"去重后剩余记录数: {len(df_deduplicated)}")

    df_deduplicated.to_csv(args.output_file, index=False)
    logger.debug(f"去重后的数据已保存到: {args.output_file}")

if __name__ == "__main__":
    main()