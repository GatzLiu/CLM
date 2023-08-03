

dataset = 1         # 0:Amazon, 1:KuaiRand
validate_test = 0   # 0:Validate, 1: Test

DATASET = ['Tamll', 'KuaiRand'][dataset]
MODEL = ['CLM', 'NCF', 'NGCF', 'LightGCN', 'LGCN'][0]
CANDIDATE_ITEM_LIST_LENGTH = 100
EMB_DIM = 64
BATCH_SIZE = 10000
TEST_USER_BATCH = {'Tamll': 4096, 'KuaiRand': 4096}[DATASET]
SAMPLE_RATE = 1
N_EPOCH = 200
TOP_K = [10, 20, 50, 100]
DIR = './dataset/'+DATASET+'/'
GPU_INDEX = "0"

# model save

#pred
PRED_BATCH_SIZE = 10000