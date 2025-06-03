import linktransformer as lt
import numpy as np
import pandas as pd
import os
import json

df=pd.read_csv("primary.csv")
print("Available columns in the CSV file:", df.columns.tolist())
print("\n原始数据行数:", len(df))

df_dedup=lt.dedup_rows(df,on="NAME",model="sentence-transformers/all-MiniLM-L6-v2",cluster_type= "agglomerative",
    cluster_params= {'threshold': 0.55})
print("\n去重后数据行数:", len(df_dedup))
print("\n去重后减少的行数:", len(df) - len(df_dedup))


# print("\n去重后的数据前10行:")
# print(df_dedup.head(10))


duplicates = df[~df.index.isin(df_dedup.index)]


duplicates_dict = duplicates[['ID', 'NAME']].to_dict('records')


with open("./duplicate_delete.json", "w", encoding='utf-8') as f:
    json.dump(duplicates_dict, f, ensure_ascii=False, indent=4)

print(f"\n已将被删除的{len(duplicates_dict)}条重复数据保存到 duplicate_delete.json")

df_dedup.to_csv("primary_deduplicated.csv", index=False)
print("\n去重后的数据已保存到 'primary_deduplicated.csv'")

