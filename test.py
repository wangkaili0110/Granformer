import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

# 获取最大值的test_acc
def get_test_Maxacc(data_path):
    # 从日志文件中读取数据
    df = pd.read_csv(data_path, delimiter=',', header=None, names=['Epoch', 'Loss', 'Acc', 'Avg Acc'])  # Type
    # 提取中Type中为test的数据
    df = df[df.Epoch.str.contains('^Test')]  # df.Type.str.contains('^Test')
    # 提取test acc的数据
    # 使用正则表达式和str.extract()方法提取数据
    df['Acc'] = df['Acc'].str.extract(r'test acc: (\d+\.\d+)')
    # 将列的数据类型更改为float
    df['Acc'] = df['Acc'].astype(float)
    acc = df['Acc']

    test_acc = round(acc.max()*100, 2)
    return test_acc  # 返回序列中的最大值
# 获取test_acc
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
# 绘制柱状图
def bar(x,y,y_step, color_list,):
    # 创建柱状图
    plt.bar(x, y, width=0.4, color=color_list)
    # 设置y轴刻度
    plt.ylim(round(min(y), 1) - 0.1, max(y) + 0.1)  # int(min(y))  92.4
    # 设置y轴刻度间隔
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_step))
    # 添加标题和标签
    # plt.title('The test accuracy of granulated ungranulated experiment under different algorithms',fontsize=20)
    # plt.tick_params(pad=1)
    plt.xlabel('Methods for relation characterization', fontsize=14, labelpad=6)
    plt.ylabel('Acc(%)', fontsize=14, labelpad=6)
    plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1)

    # 显示图形
    plt.show()
def get_x_ydata(data,y_setp):
    y1 = []
    x1 = []
    x2, y2 = get_x_y_peak(data)
    for a,b in zip(data.Epoch, data.Acc):
        if (a % y_setp == 0) & (a != 1000):
            y1.append(b)
            x1.append(a)
        # if((b==y2)&(a % y_setp !=0 )):
        #     y1.append(b)
        #     x1.append(a)
    return x1, y1


# 获取顶峰值的x坐标
def get_x_y_peak(data):
    y = data['Acc'].max()
    max_Acc_row = data.loc[data['Acc'].idxmax()]
    x = max_Acc_row['Epoch']
    return x, y

# 绘制折线图
def Linear(yl, labels, setp):

    x = []
    y = []
    y_peak = []
    x_peak = []
    for yy in yl:
        x1, y1 = get_x_ydata(yy, setp)
        y.append(y1)
        x.append(x1)
        # y_peak.append(yy['Acc'].max())
        # max_Acc_row = yy.loc[yy['Acc'].idxmax()]
        # x_peak.append(max_Acc_row['Epoch'])

    # for lb, xi, yi, yp, xp in zip(labels, x, y, y_peak, x_peak):
    for lb, xi, yi in zip(labels, x, y):
        plt.plot(xi, yi, label=lb)
        # print('{}的峰值的坐标是{},{}:'.format(lb, xp, yp))
       # plt.annotate(f'{yp}', xy=(xp, yp), xytext=(-30, 50), textcoords='offset points')
        # 获取折线的颜色
        line_color = plt.gca().lines[-1].get_color()
        # draw_spot(xp, yp, line_color)
        #plt.scatter(get_max_x(y_peak,yy),y_peak,color = 'red' ,marker='^', label=f'Peak: {y_peak}')
# 添加图例
    plt.legend(loc='lower right')
    # 设置y轴刻
    plt.ylim(50, 95)
    plt.xlim(0, 600)
    # 设置x轴刻度间隔
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
    # 添加标题和标签
    plt.tight_layout()

    plt.xlabel('Epochs', fontsize=10, labelpad=1)
    plt.ylabel('Acc(%)', fontsize=10, labelpad=1)
    plt.grid(axis='y')

    # 显示图形
    plt.show()

def draw_spot(x0, y0, cl):
    plt.scatter(x0, y0, s=30, color=cl)
    # plot参数中左侧[]指x的范围，右侧[]指y的范围

    show_spot = '(' + str(x0) + ',' + str(y0) + ')'
    plt.annotate(f'{y0}', xy=(x0, y0), xytext=(x0, y0))

if __name__ == '__main__':
    jia_path = '/home/oem/fsdownload/3Dpointcloud/nothing/chengfa.log'  # 加法 jiafa
    chengfa_path = '/home/oem/fsdownload/3Dpointcloud/convsampling/chengfa.log'  # 乘法 chengfa
    # nothing convsampling softsampling kernelsampling
    lapulasi_path = '/home/oem/fsdownload/3Dpointcloud/softsampling/chengfa.log'  # 拉普拉斯 lapulasi
    gaosi_path = '/home/oem/fsdownload/3Dpointcloud/kernelsampling/chengfa.log'  # 高斯 gaosi
    duoyuanerci_path = '/home/oem/fsdownload/3Dpointcloud/shangtangsampling/chengfa.log'  # 多元二次 duoyuanerci
    # linyu_path = '/home/oem/fsdownload/3Dpointcloud/nothing/chengfa.log'  # 领域 linyu
    # pianxu_path = '/home/oem/fsdownload/3Dpointcloud/nothing/chengfa.log'  # 偏序 pianxu

# 绘制柱形图的数据
#     jia_acc = get_test_Maxacc(jia_path)
#     chengfa_acc = get_test_Maxacc(chengfa_path)
#
#     lapulasi_acc = get_test_Maxacc(lapulasi_path)
#     gaosi_acc = get_test_Maxacc(gaosi_path)
#     duoyuanerci_acc = get_test_Maxacc(duoyuanerci_path)
#     linyu_acc = get_test_Maxacc(linyu_path)
#     pianxu_acc = get_test_Maxacc(pianxu_path)
#
#     # 示例数据
#     x = ['Add', 'Mul', 'Laplacian ', 'Gaussian ', 'Quadratic', 'Neighbor', 'Partial']  # 'Add' 'Addition',
#     y = [jia_acc, chengfa_acc, lapulasi_acc, gaosi_acc, duoyuanerci_acc, linyu_acc, pianxu_acc]  # jia_acc
#     y_step = 0.1
#     # 设置颜色
#     color_list = ['#F7C679', '#F7C679', '#618CAC', '#618CAC', '#618CAC', '#618CAC', '#618CAC']
#     # 画柱状图
#     bar(x, y, y_step, color_list)
    #画折线图

    # #获取数据
    jf_acc = get_test_acc(jia_path)
    cf_acc = get_test_acc(chengfa_path)
    lap_acc = get_test_acc(lapulasi_path)
    gs_acc = get_test_acc(gaosi_path)
    dyec_acc = get_test_acc(duoyuanerci_path)
    # ly = get_test_acc(linyu_path)
    # px_acc = get_test_acc(pianxu_path)

    # 'Addition', 'Multiplication', 'Laplacian', 'Gaussian', 'Multivariate quadratic', 'Neighborhood', 'Partial'
    # 'Nonlinearization', 'convolution linearization', 'Soft linearization', 'Granformer'
    labels = ['Nonlinearization', 'Soft linearization', 'Granformer']
    Linear([jf_acc, lap_acc, gs_acc], labels, 2)  # jf_acc, cf_acc, lap_acc, gs_acc, dyec_acc, ly, px_acc
    #Linear(jf_acc,[jf_acc,dyec_acc],'Addition',1)





