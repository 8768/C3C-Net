import argparse
import matplotlib.pyplot as plt
import numpy as np
from libcity.utils.visualize import VisHelper

def visualize_result(files, nodes_id, time_se, visualize_file):
    # file_obj = h5py.File(h5_file, "r") # 获得文件对象，这个文件对象有两个keys："predict"和"target"
    file = np.load(files['momo'])

    astgcn = np.load(files['astgcn'])
    geml = np.load(files['geml'])
    cstn = np.load(files['cstn'])

    # cstn_target = cstn['truth'][:, 0, :, :, :, :, 0]
    # cstn_target = cstn_target.reshape((3504,75,75))
    cstn_prediction = cstn['prediction'][8:, 0, :, :, :, :, 0]
    cstn_prediction = cstn_prediction.reshape((280, 100, 100))
    # cstn_prediction = cstn_prediction.reshape((3504, 75, 75))

    target = file['truth'][:, 0, :, :, 0]

    ours = file['prediction'][:, 0, :, :, 0]
    astgcn = astgcn['prediction'][:, 0, :, :, 0]
    geml = geml['prediction'][:, 0, :, :, 0]

    sum = np.sum(target, axis=0)
    # file_obj = gt_file
    # prediction = file_obj["predict"][:][:, :, 0]  # [N, T],切片，最后一维取第0列，所以变成二维了，要是[:, :, :1]那么维度不会缩减
    # target = file_obj["target"][:][:, :, 0]  # [N, T],同上
    # file_obj.close()
    row = 56
    col = 55
    lenth = 288 # 6day
    end = len(target)
    start = end - lenth
    ours = ours[start:end, row, col]
    astgcn = astgcn[start:end, row, col]
    geml = geml[start:end, row, col]
    # cstn_target = cstn_target[:, row, col]
    cstn_prediction = cstn_prediction[:, row, col] # [T1]，将指定节点的，指定时间的数据拿出来


    plot_target = target[start:end, row, col]   # [T1]，同上

    plt.figure()
    plt.grid(True, linestyle="-.", linewidth=0.5)
    # plt.plot(np.array([t for t in range(len(target)-8)]), cstn_target, ls="-", marker=" ", color="k")
    plt.plot(np.array([t for t in range(lenth)]), ours, ls="-", marker=" ", color="r")
    plt.plot(np.array([t for t in range(lenth-8)]), cstn_prediction, ls="-", marker=" ", color="k")
    plt.plot(np.array([t for t in range(lenth)]), astgcn, ls="-", marker=" ", color="y")
    plt.plot(np.array([t for t in range(lenth)]), geml, ls="-", marker=" ", color="g")
    plt.plot(np.array([t for t in range(lenth)]), plot_target, ls="-", marker=" ", color="b")

    plt.legend(["ours", "cstn", "astgcn", "geml", "target"], loc="upper right")

    plt.axis([0, lenth,
              np.min(np.array([np.min(ours), np.min(plot_target)])),
              np.max(np.array([np.max(ours), np.max(plot_target)]))])

    visualize_file = visualize_file + str(row) + '_to_' + str(col)
    plt.title(visualize_file)
    # plt.show()
    plt.savefig(visualize_file + ".png")

def visualize_od(files,savepath):
    file = np.load(files['momo'])
    target = file['truth'][:, 0, :, :, 0]

    ours = file['prediction'][:, 0, :, :, 0]

    for i in range(224,225):
        fig = plt.figure(figsize=(10, 8))
        sub1 = fig.add_subplot(1, 1, 1)



        # 定义横纵坐标的刻度
        # ax.set_yticks(range(len(yLabel)))
        # ax.set_yticklabels(yLabel, fontproperties=font)
        # ax.set_xticks(range(len(xLabel)))
        sub1.set_xlabel('6:30 p.m.')
        # 作图并选择热图的颜色填充风格，这里选择hot
        t = target[-i,:,:]
        im = sub1.imshow(target[-i,:,:], cmap='GnBu')
        # sub1.set_title('target')
        # sub1.colorbar(im)
        # sub2 = fig.add_subplot(1, 2, 2)
        # sub2.imshow(ours[-i, :, :], cmap='GnBu')
        # sub2.set_title('predict')
        # 增加右侧的颜色刻度条
        plt.colorbar(im)
        # 增加标题
        title = "{}.png".format(i)
        # plt.title(title)
        savepath = 'time/cd_compare/predict6pm.png'
        plt.savefig(savepath)
        # show
        plt.show()

if __name__ == '__main__':
    files = {}
    files['astgcn'] = './libcity/cache/astgcn/evaluate_cache/2022_12_19_18_23_21_ASTGCNCommon_CD-TAXI_predictions.npz'
    files['momo'] ='./libcity/cache/new_cd_base/evaluate_cache/2023_02_22_20_31_08_MOMO_CD-TAXI_predictions.npz'
    files['cstn'] = './libcity/cache/cstn-cd/evaluate_cache/2022_12_17_23_23_56_CSTN_CD-TAXI_predictions.npz'
    files['geml'] = './libcity/cache/geml-cd/evaluate_cache/2022_12_17_20_43_22_GEML_CD-TAXI_predictions.npz'

    # files['astgcn'] = './libcity/cache/ASTGCN-NYC/evaluate_cache/2023_02_20_21_09_04_ASTGCNCommon_NYC_TOD_predictions.npz'
    # files['momo'] = './libcity/cache/new_nyc_base/evaluate_cache/2023_02_23_07_15_18_MOMO_NYC_TOD_predictions.npz'
    # files['cstn'] = './libcity/cache/CSTN-NYC/evaluate_cache/2023_02_18_23_16_37_CSTN_NYC_TOD_predictions.npz'
    # files['geml'] = './libcity/cache/GEML-NYC-out1/evaluate_cache/2023_02_21_11_24_59_GEML_NYC_TOD_predictions.npz'

    save_name = 'visualized_cd_'
    visualize_od(files,save_name)
    # visualize_result(files, 2, 1, save_name)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str,
    #                     default='Seattle', help='the name of dataset')
    # parser.add_argument('--save_path', type=str,
    #                     default="./visualized_data/", help='the output path of visualization')
    #
    # args = parser.parse_args()
    #
    # helper = VisHelper(vars(args))
    # helper.visualize()
