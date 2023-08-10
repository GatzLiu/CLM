from params.params_common import *

LR = [0.001, 0.0][dataset]
LAMDA = [0.2, 0.01][dataset]
alpha_ltr = 1.0
alpha_wtr = 1.0
alpha_cmtr = 1.0
alpha_ftr = 1.0
alpha_lvtr = 1.0
beta_ltr = 1.0
beta_wtr = 1.0
beta_cmtr = 1.0
beta_ftr = 1.0
beta_lvtr = 1.0
# model save

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR, 'LAMDA': LAMDA,
            'PXTR_DIM': PXTR_DIM, 'ITEM_DIM': ITEM_DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': False,
            'TEST_VALIDATION': 'Validation', 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE,'LOSS_FUNCTION': 'CrossEntropy',
            'OPTIMIZER': 'SGD', 'SAMPLER': 'MMOE', 'AUX_LOSS_WEIGHT': 0, 'DIR': DIR, 'alpha_ltr': alpha_ltr,  'alpha_wtr': alpha_wtr,
            'alpha_cmtr': alpha_cmtr, 'alpha_ftr': alpha_ftr, 'alpha_lvtr': alpha_lvtr, 'beta_ltr': beta_ltr, 'beta_wtr': beta_wtr,
            'beta_cmtr': beta_cmtr, 'beta_ftr': beta_ftr, 'beta_lvtr': beta_lvtr}
