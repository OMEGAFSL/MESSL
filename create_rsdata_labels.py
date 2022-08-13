import pandas as pd
import os


data_root = 'data'

UCM_path = os.path.join(data_root, 'UCM')
NWPU45_path = os.path.join(data_root, 'NWPU45')
WHU19_path = os.path.join(data_root, 'WHURS19')

rs_data_subset = ['train', 'val', 'test']

# 根据数据集的划分，生成csv文件
def get_rs_data_csv(data_path, subset):
    for sub in subset:
        cls_list = os.listdir(os.path.join(data_path, sub))  # 列出所有文件（包括文件夹）
        img_path = []
        img_label = []
        cls_name = []
        for c, cls in enumerate(cls_list):
            img_list = os.listdir(os.path.join(data_path, sub, cls))  # 某类的所有数据
            for i in range(len(img_list)):
                img_path.append(os.path.join(data_path, sub, cls, img_list[i]))
                img_label.append(c)
                cls_name.append(cls)
        data = [img_path, img_label, cls_name]
        df_data = pd.DataFrame(data)
        print(df_data.size)
        df_data.to_csv(os.path.join(data_path, '{}.csv'.format(sub)))


# 测试是否划分成功
def test(data_path, subset):
    f_csv = pd.read_csv(os.path.join(data_path, '{}.csv'.format(subset)))
    l = list(f_csv.loc[0])[1::] #
    print(l)

# 统计每类有多少图
def whurs19_image_count():
    for sub in rs_data_subset:
        cls_list = os.listdir(os.path.join(WHU19_path, sub))  # 列出所有文件（包括文件夹）
        for c, cls in enumerate(cls_list):
            img_list = os.listdir(os.path.join(WHU19_path, sub, cls))  # 某类的所有数据
            print('{} : {}'.format(cls, len(img_list)))

# 统一化每类数据的数量为num，抛弃多余的
def get_whurs19_data_csv(num=50):
    for sub in rs_data_subset:
        cls_list = os.listdir(os.path.join(WHU19_path, sub))  # 列出所有文件（包括文件夹）
        img_path = []
        img_label = []
        cls_name = []
        for c, cls in enumerate(cls_list):
            img_list = os.listdir(os.path.join(WHU19_path, sub, cls))  # 某类的所有数据
            img_list = img_list[:num] # 取前num个
            for i in range(len(img_list)):
                img_path.append(os.path.join(WHU19_path, sub, cls, img_list[i]))
                img_label.append(c)
                cls_name.append(cls)
        data = [img_path, img_label, cls_name]
        df_data = pd.DataFrame(data)
        print(df_data.size)
        df_data.to_csv(os.path.join(WHU19_path, '{}_count50.csv'.format(sub)))


if __name__ == '__main__':
    print(os.getcwd()) # 打印当前运行目录
    get_rs_data_csv(NWPU45_path, rs_data_subset)
    get_rs_data_csv(UCM_path, rs_data_subset)
    get_whurs19_data_csv()
    # test(NWPU45_path, rs_data_subset[1])
    # whurs19_image_count()