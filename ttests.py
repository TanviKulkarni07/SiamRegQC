from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

def compare_algorithms(df):
    
    ttest_results = {'Agorithm':[],
                     'mean':[],
                     'std':[],
                    't-stat': [],
                    'p-value': []}
    group2 = np.array(df['Ground-Truth'])
    
    for col in ['Feature-Based', 'Intensity-Based', 'FFT-Based']:
        group1 = np.array(df[col])
        t_statistic, p_value = ttest_ind(group1, group2, equal_var=False)
        ttest_results['Agorithm'].append(col)
        ttest_results['mean'].append(group1.mean())
        ttest_results['std'].append(group1.std())
        ttest_results['t-stat'].append(t_statistic)
        ttest_results['p-value'].append(p_value)
    return pd.DataFrame(ttest_results)

def remove_outliers(df, column_name, lower_bound=0.05, upper_bound=0.95):
    q1 = df[column_name].quantile(lower_bound)
    q3 = df[column_name].quantile(upper_bound)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return df_filtered
def compare_algorithms_Elastic(df, filename):
    
    ttest_results = {'Value': [], 'T-stat': [], 
                    'p-val': [], 'mean': [], 'std': []}
    print('-----------------Running T-Test-------------------')
    for metric in ['MSE', 'SSIM', 'SIAM']:
        for type in ['Deformed', 'ANTS', 'VXM_MSE1', 'VXM_NCC', 'VXM_Simonovosky', 'VXM_DeepSIM', 'VXM_SIAM', 'VXM_SIAM_PRO']:
            df = remove_outliers(df, f'{type}_{metric}')
            group2 = np.array(df[f'GT_{metric}'])
            group1 = np.array(df[f'{type}_{metric}'])
            t_statistic, p_value = ttest_ind(group1, group2, equal_var=False)
            ttest_results['Value'].append(f'{type}_{metric}')
            ttest_results['T-stat'].append(t_statistic)
            ttest_results['p-val'].append(p_value)
            ttest_results['mean'].append(group1.mean())
            ttest_results['std'].append(group1.std())

    ttest_df = pd.DataFrame(ttest_results)
    print(f'Saving to {filename}')
    # ttest_df.to_csv(filename, index=False)
    return ttest_df

def Welch_Ttest(df, filename):
    
    ttest_results = {'Value': [], 'T-stat': [], 
                    'p-val': [], 'mean': [], 'std': []}
    print('-----------------Running T-Test-------------------')
    # print(df.columns)
    for col in df.columns:
        if col =='Section' or col.startswith('GT'):
            continue
        
        metric = col.split('_')[-1]
        # print(col, col.split('_'), metric)
        df = remove_outliers(df, col)
        group2 = np.array(df[f'GT_{metric}'])
        group1 = np.array(df[col])
        t_statistic, p_value = ttest_ind(group1, group2, equal_var=False)
        ttest_results['Value'].append(col)
        ttest_results['T-stat'].append(t_statistic)
        ttest_results['p-val'].append(p_value)
        ttest_results['mean'].append(group1.mean())
        ttest_results['std'].append(group1.std())

    ttest_df = pd.DataFrame(ttest_results)
    print(f'Saving to {filename}')
    ttest_df.to_csv(filename, index=False)
    return ttest_df

def main():

    # df_file = '../metric_csv_files/MRI_GuysT1__20240309-215304_BS32_EP10IXI030-Guys-0708-T1050pro.csv'
    # df_file = '../metric_csv_files/MRI_GuysT1_NL_20240309-223041_BS32_EP10IXI030-Guys-0708-T1050pro.csv'
    df_file = '../metric_csv_files/MRI_GuysT1_Simon_NPNL_20240310-132842_BS32_EP10IXI030-Guys-0708-T1050pro.csv'
    # df_file = '../metric_csv_files/IXI030-Guys-0708-T1050_multimodal_metrics.csv'
    df = pd.read_csv(df_file)
    ttest_df = compare_algorithms_Elastic(df, df_file.replace('metric_csv_files', 'ttest_csv_files'))
    # ttest_df = Welch_Ttest(df, df_file.replace('metric_csv_files', 'ttest_csv_files'))
    print(ttest_df)

if __name__ == '__main__':
    main()