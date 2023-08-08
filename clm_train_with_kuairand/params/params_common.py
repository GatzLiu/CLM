

dataset = 1         # 0:Amazon, 1:KuaiRand
validate_test = 0   # 0:Validate, 1: Test

DATASET = ['Tamll', 'KuaiRand'][dataset]
MODEL = ['CLM', 'NCF', 'NGCF', 'LightGCN', 'LGCN'][0]
CANDIDATE_ITEM_LIST_LENGTH = 100
PXTR_DIM = 16
ITEM_DIM = 64
PXTR_BINS = 10000

N_EPOCH = 5
BATCH_SIZE = 2000
TEST_USER_BATCH = {'Tamll': 4096, 'KuaiRand': 4096}[DATASET]
SAMPLE_RATE = 1
TOP_K = [10, 20, 50, 100]
DIR = '../clm_pretraining_with_kuairand/dataset/'+DATASET+'/'
GPU_INDEX = "0"

# model save

#pred
PRED_BATCH_SIZE = 2000