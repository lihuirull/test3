# !/usr/bin/python 
# -*- coding: utf-8 -*-
# Author:lihuiru
# Created on 2023/11/29 9:49
from collections import defaultdict
from itertools import product

import pandas as pd
import re
import ast
import os
import numpy as np
pd.set_option('Max_columns', None)

HA_TYPES = [f"H{i}" for i in range(1, 17) if i != 3]
NA_TYPES = [f"N{i}" for i in range(1, 10) if i != 2]


# data = data[data["Protein Type"] != "combination"]
def split_residues(residues):
    """
    将残基字符串分割成位置和氨基酸部分。

    参数:
    - residues (str): 需要处理的残基字符串。

    返回:
    - List of tuples: 每个元组包含位置和氨基酸。
    """
    parts = []
    current_part = ''
    for char in residues:
        if char.isdigit() and current_part.isalpha():
            parts.append(current_part)
            current_part = char
        elif char.isalpha() and current_part.isdigit():
            parts.append(current_part)
            current_part = char
        else:
            current_part += char
    parts.append(current_part)  # 添加最后一部分
    return parts


def process_combinations_v2(residues, protein, index, *other_data):
    """
    处理残基中的单个蛋白的组合或者单个蛋白多种deletion构成的组合（用 '&' 分隔），并据此创建新的行。

    参数:
    - residues (str): 需要处理的残基字符串。
    - protein (str): 蛋白质类型。
    - index (int): 行索引。

    返回:
    - List of tuples: 每个元组包含处理后的 'Protein Type' 和 'Amino acid site'。
    """
    # 将残基分割成单独的部分
    split_residues_list = residues.split('&')
    new_rows = []
    # 遍历每个部分并创建新行
    # print(residues)
    for idx, residue in enumerate(split_residues_list):
        if "Deletion" in residue :
            deletion = residue.split("-", 1)[1]
            deletion = process_deletion(deletion)
            new_rows.append((f"{protein}-combination_{index}_{idx}", deletion, f"{protein}", *other_data))
            continue
        # elif "Deletion" in residue:
        #     deletion = process_deletion(residue)
        #     new_rows.append((f"{protein}-combination_{index}_{idx}", deletion, f"{protein}", *other_data))
        #     continue
        position = ''.join(filter(str.isdigit, residue)).strip()
        # 找到字符串中所有的数字并作为分割点
        split_points = [match.start() for match in re.finditer(r'\d', split_residues_list[-1])][-1] + 1

        amino_acids = split_residues_list[-1][split_points:]
        # 更新正则表达式，避免在匹配单个字符时产生空字符串
        updated_pattern = r"\[(.*?)\]|([^\[\]])"

        # 使用更新后的正则表达式进行拆分
        updated_split_result = [match for group in re.findall(updated_pattern, amino_acids)
                                for match in group if match]

        if '[' in amino_acids:
            for i in updated_split_result[idx]:
                for j in i:
                    # 打印每个变异
                    new_rows.append(
                        (f"{protein}-combination_{index}_{idx}", f"{position}{j}", f"{protein}", *other_data))

        else:
            # print('-'*50)

            # print(amino_acids)
            # print(idx)
            new_rows.append(
                (f"{protein}-combination_{index}_{idx}", f"{position}{amino_acids[idx]}", f"{protein}", *other_data))

    return new_rows


def process_deletion(residues):
    """
    处理删除类型的残基。

    参数:
    - residues (str): 需要处理的残基字符串。

    返回:
    - 经处理的残基数据。
    """
    positions = re.findall("\d+", residues)
    if "Complete" in residues or "Any" in residues:
        # 如果是完整删除或任意删除
        position_range = list(range(int(positions[0]), int(positions[1]) + 1))
        residues = [f"{i}-" for i in position_range]
        print(residues)
    else:
        # 如果不是完整删除或任意删除
        residues = f"{positions[0]}-"

    return residues


def process_human_residues_v3(row, index, other_columns):
    """
    根据更新的规则处理 'Amino acid site' 列，并保留其他列的内容。

    参数:
    - row (Series): 数据帧中的一行。
    - index (int): 行的索引。
    - other_columns (list): 需要保留的其他列的列名。

    返回:
    - List of tuples: 每个元组包含处理后的行信息。
    """
    protein = row['Protein Type']
    residues = row['Amino acid site']
    processed_rows = []

    # 获取其他列的数据
    other_data = {col: row[col] for col in other_columns}
    # 首先处理组合
    # print(residues)
    if '&' in residues:
        # 前面两种都是有“-”的
        if protein == "combination":
            processed_rows.extend(process_combinations_v4(residues, protein, index, *other_data.values()))
        elif protein in [f"H{i}" for i in range(1, 19)] and 'HA2' in residues:
            processed_rows.extend(process_combinations_v4(residues, protein, index, *other_data.values()))
        # 此类为同一种蛋白的多个标志物构成，也不存在‘-’，唯一存在‘-’的可能是多个不同的deletion
        else:
            processed_rows.extend(process_combinations_v2(residues, protein, index, *other_data.values()))
    else:
        # 处理含有 '[]' 的残基
        if '[' in residues:
            prefix, rest = residues.split('[')
            variants = rest.split(']')[0]
            suffix = rest.split(']')[1] if ']' in rest else ''
            for variant in variants:
                new_residue = f"{prefix}{variant}{suffix}"
                processed_rows.append((protein, new_residue, protein, *other_data.values()))
        else:
            # 处理删除类型的残基
            if "Deletion" in residues:
                if "Any" in residues:
                    residues = process_deletion(residues)
                    for single_deletion in residues:
                        processed_rows.append((protein, single_deletion, protein, *other_data.values()))
                else:
                    residues = process_deletion(residues)
            processed_rows.append((protein, residues, protein, *other_data.values()))

    return processed_rows


def process_combinations_v4(residues, protein, index, *other_data):
    """
    处理残基中的combination组合和HA亚型的组合，因为只有这两种会存在-以表明是什么蛋白或者是HA2
    （用 '&' 分隔），并据此创建新的行。

    参数:
    - residues (str): 需要处理的残基字符串。
    - protein (str): 蛋白质类型。
    - index (int): 行索引。

    返回:
    - List of tuples: 每个元组包含处理后的 'Protein Type' 和 'Amino acid site'。
    """
    # 将残基分割成单独的部分
    split_residues_list = residues.split('&')

    new_rows = []
    # 遍历每个部分并创建新行
    for idx, residue in enumerate(split_residues_list):
        residue = residue.strip()
        protein_type = residue.split("-")[0]
        if "Deletion" in residue:
            deletion = residue.split("-", 1)[1]
            deletion = process_deletion(deletion)
            new_rows.append((f"{protein}-combination_{index}_{idx}", deletion, f"{protein_type}", *other_data))
            continue
        position = ''.join(filter(str.isdigit, residue.split('-')[1])).strip()
        split_points = [match.start() for match in re.finditer(r'\d', residue)][-1] + 1
        amino_acids = residue[split_points:]

        if '[' in amino_acids:
            amino_acid = amino_acids.split("[")[1].split("]")[0]
            for i in amino_acid:
                new_rows.append(
                    (f"{protein}-combination_{index}_{idx}", f"{position}{i}", f"{protein_type}", *other_data))
        else:
            new_rows.append(
                (f"{protein}-combination_{index}_{idx}", f"{position}{amino_acids}", f"{protein_type}", *other_data))

    return new_rows


def merge_dicts_with_list(dict_list):
    """
    合并字典列表的函数，如果键重复，则将值合并为列表。

    参数:
    - dict_list (list): 包含字典的列表。

    返回:
    - 合并后的字典。
    """
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key in merged_dict:
                # 如果键已存在，则合并值为列表
                if not isinstance(merged_dict[key], list):
                    merged_dict[key] = [merged_dict[key]]
                merged_dict[key].append(value)
            else:
                # 如果键不存在，则直接添加
                merged_dict[key] = value
    return merged_dict


def generate_combinations(group):
    """
    根据特定类型对变异进行分组，并生成所有可能的组合。

    参数:
    - group (DataFrameGroupBy): 分组后的DataFrame。

    返回:
    - 所有可能的组合列表。
    """
    # 根据特定类型对变异进行分组
    spec_type_groups = group.groupby('Specific Type')
    # 创建字典以存储按特定类型映射到蛋白质的变异列表
    mutation_dict = defaultdict(list)
    for spec_type, g in spec_type_groups:
        for _, row in g.iterrows():
            if type(row["Mutation"]) == str:
                mutation_dict[spec_type].append({row['Protein']: row['Mutation'].strip()})
            else:
                mutation_dict[spec_type].append({row['Protein']: row['Mutation']})
    # 提取每个键对应的字典列表
    values_lists = [mutation_dict[key] for key in mutation_dict]
    # 生成所有可能的组合并合并字典
    combinations = [merge_dicts_with_list(comb) for comb in product(*values_lists)]
    return combinations


def read_and_prepare_data(file_name, column_names):
    """读取Excel文件并进行初步处理，包括调整列名"""
    data = pd.read_excel(file_name, engine = "openpyxl")
    # Replace empty strings with NaN
    data.replace('', np.nan, inplace = True)
    data = data.dropna(subset = ["Protein Type"],how = "all", axis = 0)
    data = data.dropna(how = "all", axis = 1)
    data.columns = column_names + data.columns[len(column_names):].tolist()
    data["Amino acid site"] = data["Amino acid site"].str.split('(', expand = True)[0]
    return data


def process_and_group_data(data, output_file):
    """处理数据并分组"""
    other_columns = data.columns.difference(['Protein Type', 'Amino acid site'])
    processed_data = []
    for index, row in data.iterrows():
        processed_rows = process_human_residues_v3(row, index, other_columns.tolist())
        processed_data.extend(processed_rows)

    column_names = ["Protein Type", "Mutation", "Protein"] + other_columns.tolist()
    processed_df = pd.DataFrame(processed_data, columns = column_names)

    processed_df.to_csv(output_file, index = False)

    processed_df["Specific Type"] = processed_df["Protein Type"].str.rsplit("_", n = 1).str[-1]
    processed_df['Protein Type'] = processed_df['Protein Type'].str.replace(r'_\d+$', '', regex = True)
    return processed_df.groupby('Protein Type')


def generate_protein_dict(grouped_data):
    """生成蛋白质字典"""
    new_protein_dict = defaultdict(list)
    for name, group in grouped_data:
        if 'combination' in name:
            new_protein_dict[name] = generate_combinations(group)
        else:
            new_protein_dict[name].extend(
                {row['Protein']: row['Mutation'].strip() if isinstance(row['Mutation'], str) else row['Mutation']}
                for _, row in group.iterrows()
            )
    return new_protein_dict


def main(input_file, output_file):
    """主函数"""
    column_names = ['Protein Type', 'Amino acid site']  # 列名调整
    data = read_and_prepare_data(input_file, column_names)
    grouped_data = process_and_group_data(data, output_file)
    new_protein_dict = generate_protein_dict(grouped_data)

    # 可以选择打印或返回结果
    print(len(new_protein_dict.keys()))
    print(new_protein_dict)
    return new_protein_dict


# 调用主函数
# if __name__ == '__main__':
#     # phenotype = "mammalian_virulence"
#     # phenotype = "human_adaptation"
#     # phenotype = "transmissibility"
#     phenotype = "drug_resistance"
#     new_protein_dict = main(f"drug_resistance_std.xlsx",
#                             f"../../data/markers_for_extract/{phenotype}_formated.csv")
#     # new_protein_dict = main(f"../data/markers_for_extract/{phenotype}.xlsx",
#     #                         f"../data/markers_for_extract/{phenotype}_formated.csv")

t = [{'N1': '106I'}, {'N1': '219Q'}, {'N1': '36-'}, {'N1': '44Q'}, {'N1': '49-'}, {'N1': '49-'}, {'N1': '50-'},
     {'N1': '51-'}, {'N1': '52-'}, {'N1': '53-'}, {'N1': '54-'}, {'N1': '55-'}, {'N1': '56-'}, {'N1': '57-'},
     {'N1': '58-'}, {'N1': '59-'}, {'N1': '60-'}, {'N1': '61-'}, {'N1': '62-'}, {'N1': '63-'}, {'N1': '64-'},
     {'N1': '65-'}, {'N1': '66-'}, {'N1': '67-'}, {'N1': '68-'}, {'N1': '69-'}, {'N1': '70-'}, {'N1': '71-'},
     {'N1': '72-'}, {
         'N1': ['49-', '50-', '51-', '52-', '53-', '54-', '55-', '56-', '57-', '58-', '59-', '60-', '61-', '62-', '63-',
                '64-', '65-', '66-', '67-', '68-', '69-', '70-', '71-', '72-']}, {'N1': '72Q'}]


# def format_markers(value, add_protein = ''):
#     if isinstance(value, str):
#         if add_protein:
#             return f'{add_protein}-{value}'
#         else:
#             return value
#     elif isinstance(value, list):
#         # 使用列表推导式来检查每个字符串中是否含有 '-'
#         contains_dash = ['-' in s for s in value]
#
#         # 检查是否所有的字符串都包含短横线 '-'
#         all_contain_dash = all(contains_dash)
#         if all_contain_dash:
#             return value[0].split('-')[0] + '-' + value[-1].split('-')[0] + "CompleteDeletion"
#         res = ''
#         for marker in value:
#             if '-' in marker:
#                 amino_acid = marker.split('-')[0]
#                 if add_protein:
#                     res += f"{add_protein}-{amino_acid}Deletion&"
#                 else:
#                     res += f"{amino_acid}Deletion&"
#             else:
#                 if add_protein:
#                     res += f"{add_protein}-{marker}&"
#                 else:
#                     res += f"{marker}&"
#         return res
# def deal_dict(i):
#
#     if len(i) == 1:
#         value = list(i.values())[0]
#         return format_markers(value).rsplit("&",1)[0]
#     else:
#         res = ""
#         for prot, value in i.items():
#             marker_symbol = format_markers(value,prot)
#             if marker_symbol.endswith("&"):
#                 res += f"{marker_symbol}"
#             else:
#                 res += f"{marker_symbol}&"
#         return res.rsplit("&",1)[0]

def format_marker(marker, protein_prefix = ''):
    """
    格式化单个遗传标记。如果标记中包含短横线（'-'），则仅保留短横线之前的部分，并附加'Deletion'。
    如果提供了蛋白质前缀，它将被添加到标记之前。

    参数:
        marker (str): 需要格式化的遗传标记。
        protein_prefix (str): 可选，添加到每个标记之前的前缀。

    返回:
        str: 格式化后的遗传标记。
    """
    # 检查标记是否包含短横线，并相应地分割。
    if '-' in marker:
        amino_acid = marker.split('-')[0]
        deletion_suffix = "Deletion"
    else:
        amino_acid = marker
        deletion_suffix = ""

    # 组合蛋白质前缀、氨基酸和删除后缀。
    formatted_marker = f"{protein_prefix}-{amino_acid}{deletion_suffix}" if protein_prefix else f"{amino_acid}{deletion_suffix}"
    return formatted_marker


def format_marker_list(markers, protein_prefix = ''):
    """
    格式化标记列表或单个标记字符串。
    如果是列表且所有元素都包含短横线，则返回特殊格式的字符串。
    否则，列表中的每个标记都将单独格式化。

    参数:
        markers (str 或 list): 表示遗传标记的字符串或字符串列表。
        protein_prefix (str): 可选，添加到每个标记之前的前缀。

    返回:
        str: 代表格式化后的标记的单个字符串，用'&'连接。
    """
    # 如果输入是单个字符串，直接格式化。
    if isinstance(markers, str):
        return format_marker(markers)

    # 确定列表中所有标记是否都包含短横线。
    all_contain_dash = all('-' in marker for marker in markers)
    if all_contain_dash:
        # 如果所有标记都包含短横线，创建特殊格式的字符串。
        start = markers[0].split('-')[0]
        end = markers[-1].split('-')[0]
        return f"{start}-{end}CompleteDeletion"

    # 单独格式化每个标记并用'&'连接。
    return '&'.join(format_marker(marker, protein_prefix) for marker in markers)


def process_dictionary(data_dict):
    """
    处理包含遗传标记的字典。
    如果字典只有一个键值对，直接格式化该值。
    对于多个键值对，分别格式化每个键值对并用'&'连接。

    参数:
        data_dict (dict): 以蛋白质名称为键，遗传标记为值的字典。

    返回:
        str: 代表字典内容格式化后的单个字符串。
    """
    # 如果只有一个键值对，直接处理这个值。
    if len(data_dict) == 1:
        return format_marker_list(next(iter(data_dict.values())))

    # 如果有多个键值对，分别处理每个键值对。
    return '&'.join(format_marker_list(markers, protein) for protein, markers in data_dict.items())


def compare_dicts_updated(dict1, dict2):
    for key, value1 in dict1.items():
        if key not in dict2:
            return False

        value2 = dict2[key]

        if isinstance(value1, list) and isinstance(value2, list):
            # 如果两个值都是列表，检查它们是否包含相同的元素（这里不考虑顺序）

            if not set(value1).issubset(set(value2)):
                return False
        elif isinstance(value1, str) and isinstance(value2, list):
            # 如果dict1中的值是字符串，而dict2中的值是列表，则检查字符串是否在列表中
            if value1 not in value2:
                return False
        elif value1 != value2:
            # 其他情况，直接比较值
            return False

    return True


dic2 = {'H3': ['216E', '223V', '146S', '263R', '225G', '229R'], 'M1': ['43M', '215A'],
        'M2': ['82S', '24D'], 'N1(N2 numbering)': [],
        'NP': ['482S', '184K', '437T', '105V', '253I', '373T', '133L', '286A'],
        'PA': ['383D', '224S', '190S', '550L', '237E', '321N', '149S', '295P', '409S', '394D', '330I', '100V'],
        'PA-X': [], 'PB1': ['298L', '652A', '115Q', '473V', '469T', '598L', '386R', '517I', '13P'],
        'PB1-F2': ['87E', '56V'],
        'PB2': ['627E', '715N', '191E', '661A', '504V', '559T', '495V', '283M', '339K', '309D', '66M', '89V', '133V',
                '389R', '598T', '288Q', '477G', '683T', '109V', '391E', '431M']}


def map_residues_to_h3(protein, marker_dict, convert_to_h3_dict):
    """
    Maps the residue numbers for a given protein to the H3/N2 numbering system.

    Parameters:
        protein (str): The protein identifier.
        marker_dict (dict): Dictionary containing markers for various proteins.
        convert_to_h3_dict (dict): Dictionary that maps residue numbers to H3.

    Returns:
        list: A list of residues mapped to H3 numbering system.
    """
    markers = marker_dict[protein]

    # 如果markers是字符串，将其转换为只含一个元素的列表
    if isinstance(markers, str):
        markers = [markers]

    mapped_residues = []
    for marker in markers:
        # Ensure the marker is in the expected format (e.g., "12A")
        marker_match = re.fullmatch(r"(\d+)([A-Z])", marker)
        if not marker_match:
            # print(f"Warning: marker '{marker}' is not in the correct format and will be skipped.")
            continue

        position, amino_acid = marker_match.groups()
        h3_position = convert_to_h3_dict.get(position)
        if h3_position is None:
            # print(f"Warning: Position {position} does not have an H3 mapping "
            #       f"in the structure comparison file and will be skipped.")
            continue

        mapped_residues.append(h3_position + amino_acid)

    return mapped_residues


def convert_HA_residues(marker_dict, structure_folder):
    """
    Converts HA/NA residues to H3/N2 numbering.

    Parameters:
        marker_dict: Dictionary with protein types as keys and marker lists as values.
        structure_folder: Folder path where the structure mapping files are located.

    Returns:
        Updated marker_dict with HA/NA types converted to H3/N2 numbering.
    """
    updated_marker_dict = marker_dict.copy()  # Create copy
    for protein in list(marker_dict.keys()):
        if protein in HA_TYPES:
            mapping_data = pd.read_csv(f"{structure_folder}/H3_{protein}.txt", sep = "\t", header = None,
                                       names = ['H3', protein])
            convert_to_h3_dict = dict(zip(mapping_data[protein], mapping_data['H3']))

            residues = map_residues_to_h3(protein, marker_dict, convert_to_h3_dict)
            # residues = [convert_to_H3_dict.get(re.search(r"\d+", i).group()) +
            # re.search(r"[A-Z]", i).group() for i in
            #             marker_dict[protein] if convert_to_H3_dict.get(re.search(r"\d+", i).group())]
            if "H3" in updated_marker_dict:
                updated_marker_dict["H3"].extend(residues)
            else:
                updated_marker_dict["H3"] = residues
            del updated_marker_dict[protein]  # del key
        elif protein in NA_TYPES:
            if os.path.isfile(f"{structure_folder}/N2_{protein}.txt"):
                mapping_data = pd.read_csv(f"{structure_folder}/N2_{protein}.txt", sep = "\t", header = None,
                                           names = ['N2', protein])
            else:
                mapping_data = pd.read_csv(f"{structure_folder}/{protein}_N2.txt", sep = "\t", header = None,
                                           names = [protein, 'N2'])
            convert_to_n2_dict = dict(zip(mapping_data[protein], mapping_data['N2']))

            residues = map_residues_to_h3(protein, marker_dict, convert_to_n2_dict)
            if "N2" in updated_marker_dict:
                updated_marker_dict["N2"].extend(residues)
            else:
                updated_marker_dict["N2"] = residues
            del updated_marker_dict[protein]  # del key
    return updated_marker_dict


# total_markers = defaultdict(list)
# for pro, lst in new_protein_dict.items():
#     for dic in lst:
#         # 通过convert_HA_residues全部都会变成H3的，没有影响
#         total_markers[pro].append(convert_HA_residues(dic, "../data/structure"))
# print('-' * 50)
# print(total_markers)
# for i, j in total_markers.items():
#     for s in j:
#         # if (type(s) == str):
#         #     print(s)
#         # if compare_dicts_updated(s,dic2 ):
#         if s and all(s.values()):
#             markers_formated = process_dictionary(s)
#             print(markers_formated)
#             # print('-------------')
#             # print(s.values())
#             # print(s)

# res = process_dictionary(s)
# print(res)
# for s in t:
#     res = process_dictionary(s)
#     print(res)

phenotype = "drug_resistance"
# 把HA NA单个的转变为H3 N2
data = pd.read_csv(f"../../data/markers_for_extract/{phenotype}_formated.csv")
# Splitting the "Protein Type" column in data DataFrame
# Splitting the "Protein Type" column in data DataFrame

print(data)
print(data.loc[:, "Protein Type"].tolist())
# # 假设 HA_TYPES 和 NA_TYPES 是预定义的列表
# HA_TYPES.append('H3')
# NA_TYPES.append('N2')


data_with_combination = data[
    data.loc[:, "Protein Type"].isin(HA_TYPES) | data.loc[:, "Protein Type"].isin(NA_TYPES)]

data_without_combination = data[
    ~(data.loc[:, "Protein Type"].isin(HA_TYPES) | data.loc[:, "Protein Type"].isin(NA_TYPES))]
# data_with_combination.loc[:, "Residue"] = data_with_combination.loc[:, "Mutation"].apply(
#     lambda x: re.search('[A-Z-]', x).group())
# data_with_combination.loc[:, "Mutation"] = data_with_combination.loc[:, "Mutation"].apply(
#     lambda x: re.search('\d+', x).group())
print(data_with_combination)

for idx, row in data_with_combination.iterrows():
    dic = {row["Protein Type"]: row['Mutation']}
    newdic = convert_HA_residues(dic, "../../data/structure")

    # 检查 newdic 是否为空
    if list(newdic.values())[0]:
        print('-'*50)
        print(dic)
        print(newdic)
        data_with_combination.loc[idx, "Protein Type"] = list(newdic.keys())[0]
        data_with_combination.loc[idx, "Protein"] = list(newdic.keys())[0]
        data_with_combination.loc[idx, "Mutation"] = list(newdic.values())[0][0]
    else:
        # 如果 newdic 为空，可以选择跳过或赋予默认值
        # 例如：跳过当前迭代
        continue
        # 或者赋予默认值
        # data_with_combination.loc[idx, "Protein Type"] = 默认值
        # data_with_combination.loc[idx, "Mutation"] = 默认值

print(data_with_combination)

converted_data = pd.concat([data_with_combination,data_without_combination])
print(converted_data)
converted_data.to_csv(f"../../data/markers_for_extract/{phenotype}_formated.csv",index = False)