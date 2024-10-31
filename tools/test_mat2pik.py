
import scipy.io
import pickle

mat_file = 'test_out/pytorch_result_street.mat'
save_pkl = mat_file.split('.')[0]

# 从MAT文件中读取数据
mat_data = scipy.io.loadmat(mat_file)

# 将数据保存为pkl文件
with open(f'{save_pkl}.pkl', 'wb') as pkl_file:
    pickle.dump(mat_data, pkl_file)

