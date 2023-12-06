# !/usr/bin/python 
# -*- coding: utf-8 -*-
# Author:lihuiru
# Created on 2023/12/4 16:20
import pandas as pd
import re

pd.set_option("Max_columns", None)


# #1.定义耐药水平为文本标签
# # 定义将RGB转换为十六进制颜色的函数
# def rgb_to_hex(r, g, b):
#     return "{:02X}{:02X}{:02X}".format(r, g, b)
#
# from openpyxl import load_workbook
# print(rgb_to_hex(237,125,49))
# print(rgb_to_hex(128,128,128))
# from openpyxl import load_workbook
#
# def update_sheet(ws):
#     for row in ws.iter_rows(min_row=2, min_col=2, max_col=7):  # 第2列到第7列
#         for cell in row:
#             font_color = cell.font.color
#             if cell.value is not None and font_color is not None and font_color.type == 'rgb':
#                 color = font_color.rgb[2:]  # 跳过ARGB中的A部分
#                 append_text = ""
#                 if color == "FF0000":   # 红色
#                     append_text = "(high)"
#                 elif color == "FFC000" or color == "FDBF2B": # 橙色
#                     append_text = "(middle)"
#                 elif color == "000000" or color == "333333": # 黑色
#                     append_text = "(low)"
#                 elif color == "808080": # 灰色
#                     append_text = "(unknown)"
#
#                 cell.value += append_text
#             else:
#                 if cell.value is not None:
#                     append_text = "(low)"
#                     cell.value += append_text
#             # 加载Excel工作簿
# wb = load_workbook("D:\\Learning\\edegfile\\test_drug.xlsx")
#
# # 更新每个工作表
# for sheet in wb.sheetnames:
#     ws = wb[sheet]
#     update_sheet(ws)
#
# # 保存修改后的工作簿
# wb.save("D:\\Learning\\edegfile\\test_drug_modified.xlsx")

# 2.把标志物全部定格到第一个药物列
# phenotype = "drug_resistance"
#
# # phenotype = "receptor_binding_alteration"
# df = pd.read_excel(f"../../data/markers_for_extract/{phenotype}.xlsx")
#
#
# def fill_from_others(row):
#     if pd.isna(row['oseltamivir']):
#         return row[2:7].dropna().iloc[0] if not row[2:7].dropna().empty else None
#     else:
#         return row['oseltamivir']
#
#
# # 应用这个函数来更新第2列
# df['oseltamivir'] = df.apply(fill_from_others, axis = 1)
# df.rename(columns = {'oseltamivir': "Amino acid site"}, inplace = True)
# df.dropna(axis = 1, how = "all", inplace = True)
# df.to_excel("drug_resistance.xlsx", index = False)


def identify_pattern(string):
    return re.search('\d+[A-Z]', string)


def identify_residue_site(string):
    return re.search('\d+([A-Z])', string).group(1), re.search('\d+', string).group()


# 定义一个函数来移除数字前的字母
def remove_letters(cell):
    if isinstance(cell, str):
        type = '(' + re.search("(low|high|unknown|middle)", cell).group() + ")"

        cell = cell.split("(")[0]
        if "Anhui" in cell:
            mutation = identify_pattern(cell.split("-")[0])
            return f"{mutation}{type}"
        elif '/' in cell and '&' not in cell:
            # 分割字符串
            mutation_lst = cell.split('/')
            # 使用列表推导式和zip函数提取位点和残基
            sites, residues = zip(*(identify_residue_site(mutation) for mutation in mutation_lst))
            # 组合位点和残基
            res = '&'.join(residues) + ''.join(sites)
            return f"{res}{type}"

        elif '/' not in cell and "&" not in cell and '[' not in cell:
            if "Deletion" in cell:
                return f"{cell}{type}"
            return f"{''.join(filter(str.isdigit, cell)) + cell[-1]}{type}"

        elif '[' in cell:
            print(cell)
            print('res:')
            print(re.search("[0-9]+.*", cell).group())
            return f'{re.search("[0-9]+.*", cell).group()}{type}'
        else:
            return f"{cell}{type}"
    return cell


# 3.整理顶格后的第一列药物的标志物格式为标准格式
# df = pd.read_excel(r"drug_resistance.xlsx")
# df.dropna(how = "all",inplace =True)
# df.dropna(how = "all",inplace =True,axis = 1)
# print(df.columns)
# # 应用函数到指定列
# # 定义一个函数来移除数字前的字母
# def remove_letters_ori(cell):
#     if isinstance(cell, str):
#         cell = cell.split("(")[0]
#         if "Anhui" in cell:
#             mutation = identify_pattern(cell.split("-")[0])
#             return f"{mutation}"
#         elif '/' in cell and '&' not in cell:
#             # 分割字符串
#             mutation_lst = cell.split('/')
#             # 使用列表推导式和zip函数提取位点和残基
#             sites, residues = zip(*(identify_residue_site(mutation) for mutation in mutation_lst))
#             # 组合位点和残基
#             res = '&'.join(residues) + ''.join(sites)
#             return f"{res}"
#
#         elif '/' not in cell and "&" not in cell and '[' not in cell :
#             if "Deletion" in cell:
#                 return f"{cell}"
#             return f"{''.join(filter(str.isdigit, cell)) + cell[-1]}"
#         elif '[' in cell:
#             return re.search("\d+.*",cell).group()
#         else:
#             return f"{cell}"
#     return cell
# cols_to_change = ['Amino acid site']
# for col in cols_to_change:
#     df[col] = df[col].apply(remove_letters_ori)
# df = df.loc[:,["Protein Type","Amino acid site","PMID","Source"]]
# df.to_excel("drug_resistance_std.xlsx",index = False) #用此文件获得formated.csv(test_deal_markerfile.py)

# # 4.整理所有药物的标志物格式为标准格式
df = pd.read_excel(r"D:\Learning\edegfile\test_drug_modified.xlsx")
df.dropna(how = "all", inplace = True)
df.dropna(how = "all", inplace = True, axis = 1)
# 应用函数到指定列
cols_to_change = ['oseltamivir', 'zanamivir', 'peramivir', 'laninamivir', 'baloxavir', 'adamantane']
for col in cols_to_change:
    df[col] = df[col].apply(remove_letters)


# df.to_excel("test2.xlsx",index = False)


# 5.将整理后的格式把耐药水平标签拆解出来，药物融成一列
# 定义药物列
drug_columns = ['oseltamivir', 'zanamivir', 'peramivir', 'laninamivir', 'baloxavir', 'adamantane']

# 创建一个布尔索引掩码，如果任何药物列包含"&"则为True
mask = df[drug_columns].apply(lambda x: x.str.contains('&', na = False)).any(axis = 1)
print(mask)
# 基于掩码更新'Protein Type'列，仅当行索引是True时才添加'-combination_i'
df.loc[mask, 'Protein Type'] = df.loc[mask, 'Protein Type'] + '-combination_' + df.loc[mask].index.astype(str)

# 然后“熔化”DataFrame，保留所有其他列
df_melted = df.melt(id_vars = ['Protein Type', 'PMID', 'Source', 'Unnamed: 67', 'table'], var_name = 'Drug',
                    value_name = 'Mutation')

# 分离Mutation列为位点和耐药水平
df_melted['Resistance_level'] = df_melted['Mutation'].str.extract(r'\((.*?)\)')
df_melted.loc[:, 'Mutation'] = df_melted['Mutation'].str.split("(").str[0]
print(df_melted.columns)
df_melted.to_csv("test.csv", index = False)

# 6.将拆解后的df和之前的得到的format的df合并，获得能够有耐药水平和药物名称的新的df
data = pd.read_csv("../../data/markers_for_extract/drug_resistance_formated.csv")

# Condition for 'combination' in 'Protein Type' or '[' in 'Mutation'
condition_combination = (data['Protein Type'].str.contains("combination")) | (data['Mutation'].str.contains("\["))

# Select rows meeting the condition for data_combination
data_combination = data

# Select rows not meeting the condition for data_non_combination
data_non_combination = data[~condition_combination]

# Condition for 'combination' in 'Protein Type' or '[' in 'Mutation'
condition_combination = (df_melted['Protein Type'].str.contains("combination")) | (
    df_melted['Mutation'].str.contains("\["))

# Split df_melted DataFrame into two subsets
df_melted_combination = df_melted[condition_combination]
df_melted_non_combination = df_melted[~condition_combination]

# Merge the "combination" subsets
df_melted_combination.drop("Mutation", axis = 1, inplace = True)
data_combination.loc[:, "ori"] = data_combination.loc[:, "Protein Type"]
data_combination.loc[:, "Protein Type"] = data_combination.loc[:, "Protein Type"].str.rsplit("_", 1).str[0]
print(data_combination)



merged_combination = pd.merge(df_melted_combination, data_combination, how = "left",
                              on = ["Protein Type", "PMID", "Source"])


# print(merged_combination[merged_combination.loc[:,"PMID"]=="PMID:32253432"])
# print(df_melted_combination["Drug"].tolist())
# Merge the non-"combination" subsets
merged_non_combination = pd.merge(data_non_combination, df_melted_non_combination, how = "left",
                                  on = ["Protein Type", "Mutation", "PMID", "Source"])

merged_combination.loc[:, "Protein Type"] = merged_combination.loc[:, "ori"]
# Combine the results into a final DataFrame
final_df = pd.concat([merged_combination, merged_non_combination])
print(merged_combination)
print(merged_non_combination)
# Print the final DataFrame
print(final_df)
print(final_df[final_df.loc[:,"PMID"]=="PMID:32253432"])
final_df.drop(columns = ["ori", "Unnamed: 67", "table"], inplace = True)
final_df.dropna(subset=['Drug',"Resistance_level"],inplace = True)
final_df.drop_duplicates(inplace = True)
# Get current columns as a list
cols = list(final_df.columns)

# Move 'Mutation' column to the second position (index 1)
cols.insert(1, cols.pop(cols.index('Mutation')))

# Reorder the DataFrame columns
final_df = final_df[cols]

# Save the DataFrame to an Excel file
final_df.to_excel("drug_resistance_with_resistance_level.xlsx", index=False)

# Save the DataFrame to an Excel filecsv
final_df.to_csv("../../data/markers_for_extract/drug_resistance_formated.csv", index=False)