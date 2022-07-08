import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def plot_3d_fig(list_x, list_y, z_values):
    x = range(len(list_x))
    y = range(len(list_y))
    values = z_values
    xx, yy = np.meshgrid(x, y)
    x, y = xx.ravel(), yy.ravel()
    bottom=np.zeros_like(x) 
    z = values.ravel()
    width = height = 1
    fig = plt.figure()
    ax=fig.gca(projection='3d')
    ax.bar3d(x, y, bottom, width, height, z, shade=True)
    plt.xticks([index + 0.5 for index in range(len(list_x))], list_x)
    plt.yticks([index + 0.5 for index in range(len(list_y))], list_y)
    plt.xlabel(list_x.name)
    plt.xlabel(list_y.name)
    plt.show()

def draw_heatmap(dataframe):
    # sns.set()
    fig, ax = plt.subplots(figsize=(3*2.5, 3.5*2.5))
    sns.heatmap(dataframe, annot=True, cmap="YlGnBu_r")
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    # plt.xlabel("$\Delta_d$")
    # plt.ylabel("$\Delta_o$")
    plt.show()


with open('compare_results_matchingimprove1.txt', 'r') as f:
    data = []
    for line in f.readlines():
        values = line.strip().split(' ')
        values = np.array(values, dtype=float)
        data.append(values)

df = pd.DataFrame(data,
                  columns=['gt_o', 'gt_d', 'gt_x21', 'gt_y21',
                           'pred_o', 'pred_d','pred_x21', 'pred_y21',
                           'score'],
                  dtype=float)
df_sortscore = df.sort_values(by="score", ascending=False)
df_sortscore = df_sortscore[df_sortscore['score']>=0.3]
results = []
ranges_o = np.array([-0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
ranges_d = np.array([-0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
# ranges_d1 = np.array([-0.01, 0])
# ranges_d2 = np.array([-0.01, 0])
# ranges_d = np.array([-0.01, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01])
# ranges_o = np.array([-0.01, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
for i, index_d1 in enumerate(ranges_o[:-1]):
    select_d1 = df_sortscore[(df_sortscore['gt_o']<=ranges_o[i+1]) & (df_sortscore['gt_o']>ranges_o[i])]
    # print(select_o.head())
    for j, index_d in enumerate(ranges_d[:-1]):
        select_od = select_d1[(select_d1['gt_d']<=ranges_d[j+1]) & (select_d1['gt_d']>ranges_d[j])]
        text_o = '({:.1f},{:.1f}]'.format(ranges_o[i].clip(0,1), ranges_o[i+1].clip(0,1))
        text_d = '({:.1f},{:.1f}]'.format(ranges_d[j].clip(0,1), ranges_d[j+1].clip(0,1))
        print('dealing with the o ' + text_o + ' and the d ' + text_d + ' :')
        if select_od.shape[0] > 0:
            # print(select_od.head())
            mae_o = abs(select_od['gt_o'] - select_od['pred_o']).mean()
            mae_d = abs(select_od['gt_d'] - select_od['pred_d']).mean()
            num = select_od.shape[0]
            numrate = float(select_od.shape[0]) / df_sortscore.shape[0]
            
            sign_x = ((select_od['gt_o'] - 0.5) * (select_od['pred_o'] - 0.5)) > 0
            # sign_x = sign_x | (abs(select_od['gt_d1'] - 0.5) < 0.05)
            sign_x = sign_x | (abs(select_od['gt_x21']) < 0.2) | (abs(select_od['gt_x21']) > 0.8) \
                            | (abs(select_od['pred_x21']) < 0.2) | (abs(select_od['pred_x21']) > 0.8) 
            sign_y = ((select_od['gt_d'] - 0.5) * (select_od['pred_d'] - 0.5)) > 0
            # sign_y = sign_y | (abs(select_od['gt_d2'] - 0.5) < 0.05)
            sign_y = sign_y | (abs(select_od['gt_y21']) < 0.2) | (abs(select_od['gt_y21']) > 0.8) \
                            | (abs(select_od['pred_y21']) < 0.2) | (abs(select_od['pred_y21']) > 0.8)
            sign = sign_x & sign_y
            print(select_od[~sign].head())

            # if ranges_o[i+1] <= 0.5:
            #     sign = (select_od['gt_d'] - 0.5) * (select_od['pred_d'] - 0.5) > 0
            # else:
            #     sign = (select_od['gt_d'] - 0.5) * np.sign(select_od['gt_eta']) * (select_od['pred_d']  - 0.5) * np.sign(select_od['pred_eta']) > 0
            # # sign = sign | (select_od['gt_x21'] < 6) | (select_od['gt_x32'] < 6) | (abs(select_od['pred_x32']) < 6) | (abs(select_od['pred_x21']) < 6)
            # sign = sign | ((select_od['gt_x21'] / (select_od['gt_x21'] + select_od['gt_x32']))  < 1.0/6) | \
            #               ((select_od['gt_x32'] / (select_od['gt_x21'] + select_od['gt_x32']))  < 1.0/2) | \
            #               (abs(select_od['pred_x21'] / (select_od['pred_x21'] + select_od['pred_x32']))  < 1.0/6) | \
            #               (abs(select_od['pred_x32'] / (select_od['pred_x21'] + select_od['pred_x32']))  < 1.0/2)
            
            match_rate = float(sign.sum()) / num
            # match_rate = float(np.sum(sign)) / num
            dismath_rate = 1 - match_rate
        else:
            print('There is none values')
            num = 0
            numrate = 0
            mae_o = 0
            mae_d = 0
            dismath_rate = 0
        line = [text_o, text_d, num, numrate, mae_o, mae_d, dismath_rate]
        results.append(line)
df_results = pd.DataFrame(results,
                  columns=['range_o', 'range_d', 'num', 'num_rate', 'mae_o', 'mae_d', 'error_match'],
                  dtype=float)
df_results['error_match_total'] = df_results['num_rate'] * df_results['error_match']
print(df_results)
# print(df_results.sort_values(by="error_match_total"))
print(df_results.describe())

mae_o = (df_results['num_rate'] * df_results['mae_o']).sum()
mae_d = (df_results['num_rate'] * df_results['mae_d']).sum()
error_match = (df_results['num_rate'] * df_results['error_match']).sum()
print('The mae of d1 = ', mae_o)
print('The mae of d2 = ', mae_d)
print('The total match error = ', error_match)

table_numrate = df_results.pivot(index='range_o', columns='range_d', values='num_rate')
# table_numrate = df_results.pivot(index='range_o', columns='range_d', values='mae_o')
# table_numrate = df_results.pivot(index='range_o', columns='range_d', values='mae_d')
# table_numrate = df_results.pivot(index='range_o', columns='range_d', values='error_match')
# table_numrate = df_results.pivot(index='range_o', columns='range_d', values='error_match_total') / error_match

# list_o = table_numrate.index
# list_d = table_numrate.columns
# values = table_numrate.values
# plot_3d_fig(list_o, list_d, values)

draw_heatmap(table_numrate)