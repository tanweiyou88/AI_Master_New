import pandas as pd

def check_dataframes_similarity(df1, df2):
    print("\n*******Results:*******\n")
    if df1.equals(df2):
        print("The two DataFrames are equal")
    else:
        print("The two DataFrames are not equal")

# # Making dataframe 1
# df1 = pd.read_csv("D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/data/result/self_CompileMetrics/self_CompileMetrics_csvFile/Zero-DCE-Outputs-2025_01_05-16_05_44.csv")
# # Making dataframe 2
# df2 = pd.read_csv("D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/data/result/self_CompileMetrics/self_CompileMetrics_csvFile/Zero-DCE-Outputs-2025_01_05-16_07_29.csv")

df1 = pd.read_csv("C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/ZeroDCE_ori_EMP/2025_02_14-10_32_11-ZeroDCE_ori-SICE_Part1-ValLossesHistory_Copy.csv")
df2 = pd.read_csv("C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/ZeroDCE_ori_EMP/2025_02_15-01_28_21-SICE_Part1-TrainResultsPool/csvFile/2025_02_15-01_28_21-ZeroDCE_ori-SICE_Part1-ValLossesHistory.csv")


check_dataframes_similarity(df1, df2)
# print("\ndf1:\n", df1)
# print("df2:\n", df2)

