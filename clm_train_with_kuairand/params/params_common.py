

GPU_INDEX = "0"
dataset = 1         # 0:Amazon, 1:KuaiRand
model = 3

DATASET = ['Tamll', 'KuaiRand'][dataset]
MODEL = ['CLM', 'PRM', 'MLP', 'SUM', 'MUL'][model]
CANDIDATE_ITEM_LIST_LENGTH = 100
PXTR_DIM = 16
ITEM_DIM = 64
PXTR_BINS = 10000

N_EPOCH = 20
validate_test = 0   # 0:Validate, 1: Test
BATCH_SIZE = 10000
PRED_BATCH_SIZE = 2000
TEST_USER_BATCH = [1000, 1000][dataset]
SAMPLE_RATE = 1
TOP_K = [10, 20, 50, 100]
DIR = '../clm_pretraining_with_kuairand/dataset/'+DATASET+'/'
