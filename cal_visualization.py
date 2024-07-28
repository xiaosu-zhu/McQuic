import os
import numpy as np

# 定义文件夹路径
newest_path = '/ssdfs/datahome/tj24011/workspace/McQuic/results/comparision/newest'
previous_path = '/ssdfs/datahome/tj24011/workspace/McQuic/results/comparision/previous'

# 遍历各子文件夹
subfolders = ['40k', '80k', '100k']  # 根据实际情况列出所有子文件夹
mean_differences = {}

for subfolder in subfolders:
    newest_subfolder = os.path.join(newest_path, subfolder)
    previous_subfolder = os.path.join(previous_path, subfolder)

    # 获取对应的文件列表
    newest_files = sorted([f for f in os.listdir(newest_subfolder) if f.endswith('.npy')])
    previous_files = sorted([f for f in os.listdir(previous_subfolder) if f.endswith('.npy')])

    afterQ_list = []
    beforeQ_list = []
    for i in range(len(newest_files)):
        # print(newest_files[i])
        # print("afterQ" in newest_files[i])
        if "afterQ" in newest_files[i]:
            afterQ_list.append(newest_files[i])
        if "beforeQ" in newest_files[i]:
            beforeQ_list.append(newest_files[i])
    
    for idx, (after, before) in enumerate(zip(afterQ_list, beforeQ_list)):
        after = np.load(os.path.join(newest_subfolder, after))
        before = np.load(os.path.join(newest_subfolder, before))
        # print(before)
        eps = np.mean(np.abs(before - after))
    
        print("newest", idx, eps)
        
    afterQ_list = []
    beforeQ_list = []
    for i in range(len(previous_files)):
        # print(previous_files[i])
        # print("afterQ" in previous_files[i])
        if "afterQ" in previous_files[i]:
            afterQ_list.append(previous_files[i])
        if "beforeQ" in previous_files[i]:
            beforeQ_list.append(previous_files[i])
    
    for idx, (after, before) in enumerate(zip(afterQ_list, beforeQ_list)):
        after = np.load(os.path.join(newest_subfolder, after))
        before = np.load(os.path.join(newest_subfolder, before))
        eps = np.mean(np.abs(before - after))
    
        print("previous", idx, eps)
    # =======
    afterQ_list = []
    beforeQ_list = []
    for i in range(len(newest_files)):
        # print(newest_files[i])
        # print("afterQ" in newest_files[i])
        if "decoded" in newest_files[i]:
            afterQ_list.append(newest_files[i])
        if "img_feature" in newest_files[i]:
            beforeQ_list.append(newest_files[i])
    
    for idx, (after, before) in enumerate(zip(afterQ_list, beforeQ_list)):
        after = np.load(os.path.join(newest_subfolder, after))
        before = np.load(os.path.join(newest_subfolder, before))
        af_norm = np.linalg.norm(after)
        be_norm = np.linalg.norm(before)
        
        f1 = np.load(os.path.join(newest_subfolder, beforeQ_list[-1]))
        f1_norm = np.linalg.norm(f1)
        f1 = f1 / f1_norm
        after =  after / af_norm
        before = before / be_norm

        eps = np.mean(np.abs(after - f1))
    
        print("newest, after-beforeF", idx, eps)

    afterQ_list = []
    beforeQ_list = []
    for i in range(len(previous_files)):
        # print(previous_files[i])
        # print("afterQ" in previous_files[i])
        if "decoded" in previous_files[i]:
            afterQ_list.append(previous_files[i])
        if "img_feature" in previous_files[i]:
            beforeQ_list.append(previous_files[i])
    
    for idx, (after, before) in enumerate(zip(afterQ_list, beforeQ_list)):
        after = np.load(os.path.join(newest_subfolder, after))
        before = np.load(os.path.join(newest_subfolder, before))
        eps = np.mean(np.abs(before - after))
    
        print("previous, after-beforeF", idx, eps)