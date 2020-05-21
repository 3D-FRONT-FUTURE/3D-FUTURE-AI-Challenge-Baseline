'''
Compute distance for image and 3d models and generate submit txt file
'''
import numpy as np
import os
import glob
import pdb
from tqdm import tqdm

def normalize_rows(x):
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

def cos_similarity(query_feat, pool_feats):
    cos_score = np.dot(pool_feats, query_feat.T)
    cos_score = np.reshape(cos_score, cos_score.shape[0])
    top_k = np.argsort(cos_score)[::-1]
    return top_k

#pdb.set_trace()
shape_infos = os.listdir(os.path.join('dataset/test_data', 'input_data'))
if '.DS_Store' in shape_infos: shape_infos.remove('.DS_Store')
shape_infos.sort()

shapes = sorted(glob.glob('extract_workshop_baseline_notexture_3d_v1/*.npy'))
print('loading: 3d features')

feat_bank = []
shape_pool = []
for i in range(len(shapes)):
    shape_id = shapes[i].split('/')[-1][0:-4]
    feat = np.load(shapes[i], allow_pickle=True).item()
    feat_bank.append(feat['feat_query'][0,:])
    shape_pool.append(shape_id)

feat_bank = np.stack(feat_bank)
shape_num = len(feat_bank)
top_k_count = 0
pose_count = 0
cate_count = 0
center_count = 0
valid_count = 0
retrieval_results = []
eval_view = 0
count = 0
top_3_count = 0
top_10_count = 10

with open('retrieval_results.txt', 'w') as f:
    for i in tqdm(range(len(shape_infos))):
        shape_info = shape_infos[i]
        # 2d feature data
        image_name = shape_info.split('.')[0]
        
        query_path = 'extract_workshop_baseline_notexture_2d_v1/' + image_name + '.npy'
        valid_count += 1

        feat = np.load(query_path, allow_pickle=True).item()
        query_feat = feat['feat_query'][0,:]
        pred_cate = feat['pred_cate'][0,:]
        
        feat_bank_temp = feat_bank.copy()
        feat_2d_norm = normalize_rows(query_feat)
        feat_3d_norm = normalize_rows(feat_bank_temp)
        
        #pdb.set_trace()
        #top_k = cos_similarity(feat_2d_norm, feat_3d_norm)
        l2_dis = np.sqrt(np.sum((feat_3d_norm - feat_2d_norm)**2, axis=1))
        
        # collect distance 
        sample_dist = np.around(l2_dis, decimals=3)
        dist_value = ['{:.4}'.format(item) for item in sample_dist]
        #dist_value = ['{:f}'.format(item) for item in sample_dist]
        dist_str = ''
        for index, value in enumerate(dist_value):
            if index == len(dist_value) - 1:
                dist_str += value + '\n'
            else:
                dist_str += value + ', '
        
        line = image_name + ': ' + dist_str
        f.write(line)


        top_k = np.argsort(l2_dis)

        top_k = top_k[0:20]
        top_k_id = []
        for j in range(len(top_k)):
            top_k_id.append(shape_pool[top_k[j]])
        
f.close()
#np.save('top20_retrieval_workshop_baseline_v1_256.npy', retrieval_results)

