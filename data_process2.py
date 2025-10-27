import pandas as pd
import numpy as np
import re


def process_onehot(df, column_name):
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].replace({"": "无"}).fillna('无')
    onehot_df = pd.get_dummies(df_copy[column_name], prefix=column_name).astype(int)

    if f"{column_name}_无" in onehot_df.columns:
        onehot_df = onehot_df.drop(columns=[f"{column_name}_无"])

    return onehot_df


def process_multihot(df, column_name):
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].fillna('无')
    df_copy[column_name] = df_copy[column_name].replace({r'[，,+、]': ','}, regex=True)
    df_copy[column_name] = df_copy[column_name].replace({r'[吴]': '无'}, regex=True)
    multi_values = df_copy[column_name].str.split(',', expand=False)

    all_categories = set()
    for row in multi_values.dropna():
        all_categories.update([category.strip() for category in row])

    onehot_df = pd.DataFrame()
    for category in all_categories:
        onehot_df[f"{column_name}_{category}"] = multi_values.apply(
            lambda x: 1 if category in (x or []) else 0
        )

    if f"{column_name}_无" in onehot_df.columns:
        onehot_df = onehot_df.drop(columns=[f"{column_name}_无"])

    return onehot_df


def process_afp(df, column_name):
    df['AFP_less_400'] = 0
    df['AFP_greater_400'] = 0

    def clean_value(x):
        if pd.isnull(x):
            return None
        try:
            x_clean = str(x).replace(">", "").replace("<", "").strip()
            return float(x_clean)
        except Exception:
            return None

    df['AFP_clean'] = df[column_name].apply(clean_value)
    df.loc[df['AFP_clean'].notnull() & (df['AFP_clean'] < 400), 'AFP_less_400'] = 1
    df.loc[df['AFP_clean'].notnull() & (df['AFP_clean'] > 400), 'AFP_greater_400'] = 1

    df.drop(columns=['AFP_clean'], inplace=True)
    df = df[['AFP_less_400', 'AFP_greater_400']]
    return df


def process_target(df, col):
    df_copy = df.copy()
    df_copy['target'] = df_copy[col].map({2: 0, 1: 1}).fillna(-1).astype(int)
    return df_copy[['target']]


def read_process_data(filename, dict):
    df = pd.read_excel(filename)
    outputs = []
    print(df.columns)

    for key in dict.keys():
        col_list = dict[key]
        for col in col_list:
            if key == "one_hot":
                outputs.append(process_onehot(df, col))
            if key == "multi_hot":
                outputs.append(process_multihot(df, col))
            if key == "AFP":
                outputs.append(process_afp(df, col))
            if key == "target":
                outputs.append(process_target(df, col))
            if key == "cluster1" or key == "cluster2":
                # **不进行KMeans聚类，但保留原始数据**
                outputs.append(df[[col]])  # 直接加入原始数据列

    outputs = pd.concat(outputs, axis=1)
    outputs = outputs[outputs['target'] != -1]
    return outputs


if __name__ == "__main__":
    filename = '副本榆林二院.xlsx'
    dict = {
        "one_hot": ["性别", "是否合并肝炎", "是否合并肝硬化",
                    "包膜是否受侵犯",
                    "肿瘤是否巨块型分化",
                    "是否开腹手术（1=开腹，无=腹腔镜手术）", "是否大范围切除","child分期",
                    "肿瘤MVI",'术后复发（是=1，否=2）' ],
        # "multi_hot": ["烟酒史", "家族史", "是否肝胆类合并症", "是否心肺类合并症", "肿瘤部位", "肝内转移情况", "术中输血评估"],
        "AFP": ["AFP"],
        "cluster1": ["年龄__y",'ALT',	'AST',	'TBIL',	'DBIL',	'R-GGT',	'ALP',	'TP',	'ALB',	'PT',	'FIB',	'INR',	'APTT',	'RBC',	'WBC',
                     'LYMP',	'MONO',	'NEUT',	'HGB',	'PLT'],  # 不聚类，保留原始列
        "cluster2": ["肿瘤直径", "失血量", "时间段",'ALBI'
    ],  # 不聚类，保留原始列
        "target": ["是否活着"],
    }

    output = read_process_data(filename, dict)
    output.to_csv('data榆林二院.csv', index=False)
