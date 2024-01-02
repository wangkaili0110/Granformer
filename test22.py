import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

#获取最大值的test_acc
def get_test_Maxacc(data_path):
    # 从日志文件中读取数据
    df = pd.read_csv(data_path, delimiter=',', header=None, names=['Type', 'Loss', 'Acc', 'Avg Acc'])
    # 提取中Type中为test的数据
    df = df[df.Type.str.contains('^Test')]
    # 提取test acc的数据
    # 使用正则表达式和str.extract()方法提取数据
    df['Acc'] = df['Acc'].str.extract(r'test acc: (\d+\.\d+)')
    # 将列的数据类型更改为float
    df['Acc'] = df['Acc'].astype(float)
    acc = df['Acc']

    test_acc = round(acc.max()*100,1)
    return  test_acc  #返回序列中的最大值
#获取test_acc
def get_test_acc(data_path):
    # 从日志文件中读取数据
    df = pd.read_csv(data_path, delimiter=',', header=None, names=['Epoch', 'Loss', 'Acc', 'Avg Acc'])

    # 提取中Type中为test的数据
    df = df[df.Epoch.str.contains('^Test')]

    # 删除"Test "前缀并更改数据类型
    df['Epoch'] = df['Epoch'].str.replace('Test ', '').astype(int)
    # 提取test acc的数据
    # 使用正则表达式和str.extract()方法提取数据
    df['Acc'] = df['Acc'].str.extract(r'test acc: (\d+\.\d+)')
    # 将列的数据类型更改为float
    df['Acc'] = df['Acc'].astype(float)
    acc = df['Acc']
    df['Acc'] = round(acc*100, 2)
    df = df[['Epoch', 'Acc']]
    return df
#绘制柱状图
def bar(x,y,y_step, color_list,):
    # 创建柱状图
    plt.bar(x, y, width=0.4, color=color_list)
    # 设置y轴刻度
    plt.ylim(int(min(y)), max(y) + 0.1)
    # 设置y轴刻度间隔
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_step))
    # 添加标题和标签
    # plt.title('The test accuracy of granulated ungranulated experiment under different algorithms',fontsize=20)
    # plt.tick_params(pad=1)
    plt.xlabel('Relation characterization', fontsize=18, labelpad=18)
    plt.ylabel('Acc', fontsize=18)

    # 显示图形
    plt.show()
def get_x_ydata(data,y_setp):
    y1 = []
    x1 = []
    x2, y2 = get_x_y_peak(data)
    for a, b in zip(data.Epoch, data.Acc):
        if (a % y_setp == 0) & (a != 1000):
            y1.append(b)
            x1.append(a)
        if((b==y2)&(a % y_setp !=0 )):
            y1.append(b)
            x1.append(a)
    return x1, y1


# 获取顶峰值的x坐标
def get_x_y_peak(data):
    y = data['Acc'].max()
    max_Acc_row = data.loc[data['Acc'].idxmax()]
    x = max_Acc_row['Epoch']
    return x,y

#绘制折线图

def Linear(yl,labels,setp,MaxNum):

    x = []
    y = []
    y_peak = []
    x_peak = []
    for yy in yl:
        x1, y1 = get_x_ydata(yy, setp)
        y.append(y1)
        x.append(x1)
        y_peak.append(yy['Acc'].max())
        max_Acc_row = yy.loc[yy['Acc'].idxmax()]
        x_peak.append(max_Acc_row['Epoch'])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.000})
    plt.subplots_adjust(hspace=0.0001)   # adjust space between axes
    for lb, xi, yi, yp, xp in zip(labels, x, y, y_peak, x_peak):
        ax1.plot(xi, yi, label=lb)
        ax2.plot(xi, yi, label=lb)

        print('{}的峰值的坐标是{},{}:'.format(lb, xp, yp))

        # 获取折线的颜色
        line_color = plt.gca().lines[-1].get_color()
        draw_spot(xp, yp, ax1, line_color)

        # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(92, MaxNum)  # outliers only
    ax2.set_ylim(60, 91.9999999)  # most of the data 91.9999999
    ax2.set_xlim(0, 1000)

# 添加图例
    plt.legend(loc='lower right')
    # 设置y轴刻
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
    # 设置x轴刻度间隔
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=100))

    # 添加标题和标签
    plt.tight_layout()

    plt.xlabel('Epochs', fontsize=10, labelpad=6, verticalalignment='center')
    plt.ylabel('Acc', fontsize=12, labelpad=7, verticalalignment='bottom', horizontalalignment='left')
    ax1.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1)  # 设置图表中的网线为虚线 linestyle='dashed'
    ax2.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1)  # 设置图表中的网线为虚线
    ax2.spines['top'].set_linestyle('--')
    ax2.spines['top'].set_color('gray')
    ax2.spines['top'].set_alpha(0.4)
    # 显示图形
    plt.show()

def draw_spot(x0, y0,plt,cl):
    plt.scatter(x0, y0, s=30, color=cl)
    plt.annotate(f'{y0}', xy=(x0, y0), xytext=(x0, y0))

if __name__ == '__main__':
    jia_path = '/home/oem/fsdownload/3Dpointcloud/nothing/lapulasi.log'  # 加法 jiafa
    chengfa_path = '/home/oem/fsdownload/3Dpointcloud/convsampling/lapulasi.log'  # 乘法 chengfa
    # nothing convsampling softsampling kernelsampling
    lapulasi_path = '/home/oem/fsdownload/3Dpointcloud/softsampling/lapulasi.log'  # 拉普拉斯 lapulasi
    gaosi_path = '/home/oem/fsdownload/3Dpointcloud/kernelsampling/lapulasi.log'  # 高斯 gaosi
    # duoyuanerci_path = '/home/oem/fsdownload/3Dpointcloud/nothing/chengfa.log'  # 多元二次 duoyuanerci
    # linyu_path = '/home/oem/fsdownload/3Dpointcloud/nothing/chengfa.log'  # 领域 linyu
    # pianxu_path = '/home/oem/fsdownload/3Dpointcloud/nothing/chengfa.log'  # 偏序 pianxu

    # 绘制柱形图的数据
#     jia_test_acc = get_test_Maxacc(jia_path)
#     chengfa_acc = get_test_Maxacc(chengfa_path)
#
#     lapulasi_acc = get_test_Maxacc(lapulasi_path)
#     gaosi_acc = get_test_Maxacc(gaosi_path)
#     duoyuanerci_acc = get_test_Maxacc(duoyuanerci_path)
#     linyu_acc = get_test_Maxacc(linyu_path)
#     pianxu_acc = get_test_Maxacc(pianxu_path)
#
#     # 示例数据
#     x = ['Multiplication', 'Laplacian', 'Gaussian ', 'Multivariate quadratic', 'Neighborhood', 'Partial']  # 'Addition',
#     y = [chengfa_acc, lapulasi_acc, gaosi_acc, duoyuanerci_acc, linyu_acc, pianxu_acc]  # jia_test_acc,
#     y_step = 0.1
#     #设置颜色
#     color_list = ['#F7C679', '#F7C679', '#618CAC', '#618CAC', '#618CAC', '#618CAC']
    # 画柱状图
    #bar(x,y,y_step,color_list)

    #画折线图

    #获取数据
    jf_acc = get_test_acc(jia_path)
    cf_acc = get_test_acc(chengfa_path)
    lap_acc = get_test_acc(lapulasi_path)
    gs_acc = get_test_acc(gaosi_path)
    # dyec_acc = get_test_acc(duoyuanerci_path)
    # ly = get_test_acc(linyu_path)
    # px_acc = get_test_acc(pianxu_path)

    labels = ['Multiplication', 'Laplacian', 'Partial']
    # 'Addition', 'Multiplication', 'Laplacian', 'Gaussian', 'Multivariate quadratic', 'Neighborhood', 'Partial'
    # Linear([cf_acc, lap_acc, px_acc], labels, 10, 93.8)
    Linear([jf_acc, cf_acc, lap_acc, gs_acc], labels, 5, 93.8)  # jf_acc, cf_acc, lap_acc, gs_acc, dyec_acc, ly, px_acc
    # jf_acc, cf_acc, lap_acc, gs_acc, dyec_acc, ly, px_acc
    #Linear(jf_acc,[jf_acc,dyec_acc],'Addition',1)





