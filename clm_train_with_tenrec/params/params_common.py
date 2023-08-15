

GPU_INDEX = "0"
dataset = 0         # 0:KuaiRand, 1:Tenrec
model = 0

DATASET = ['Tenrec', 'KuaiRand'][dataset]
MODEL = ['CLM', 'PRM', 'MLP', 'SUM', 'MUL'][model]
CANDIDATE_ITEM_LIST_LENGTH = 100
PXTR_DIM = 16
ITEM_DIM = 64
PXTR_BINS = 10000
PXTR_LIST = ['pltr', 'pwtr', 'pftr']

N_EPOCH = 50
BATCH_SIZE = 10000
PRED_BATCH_SIZE = 2000
TEST_USER_BATCH = [1000, 1000][dataset]
TOP_K = [10, 20, 30, 40, 50]
DIR = '../clm_pretraining_with_tenrec/dataset/'+DATASET+'/'
