from params.params_common import *

LR = [0.0, 0.0][dataset]
LAMDA = [0.0, 0.0][dataset]
alpha_ltr = 1.0
alpha_wtr = 1.0
alpha_cmtr = 1.0
alpha_ftr = 1.0
alpha_lvtr = 1.0
# model save

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR, 'LAMDA': LAMDA,
            'PXTR_DIM': PXTR_DIM, 'ITEM_DIM': ITEM_DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': False,
            'TEST_VALIDATION': 'Validation', 'TOP_K': TOP_K,
            'OPTIMIZER': 'SGD', 'SAMPLER': 'MMOE', 'AUX_LOSS_WEIGHT': 0, 'DIR': DIR, 'alpha_ltr': alpha_ltr,  'alpha_wtr': alpha_wtr,
            'alpha_cmtr': alpha_cmtr, 'alpha_ftr': alpha_ftr, 'alpha_lvtr': alpha_lvtr}
