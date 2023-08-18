from params.params_common import *

LR = 0.0
# weight is used to balance the scale and importance of each xtr ranking
alpha_ltr = 0.5
alpha_wtr = 0.3
alpha_cmtr = 0.3
alpha_ftr = 0.5
alpha_lvtr = 1.0
beta_ltr = 1.0
beta_wtr = 1.0
beta_cmtr = 1.0
beta_ftr = 1.0
beta_lvtr = 1.0
# model save

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR,
            'PXTR_DIM': PXTR_DIM, 'ITEM_DIM': ITEM_DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE,
            'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'TOP_K': TOP_K, 'PXTR_LIST': PXTR_LIST, 'OPTIMIZER': 'SGD', 'DIR': DIR,
            'alpha_ltr': alpha_ltr,  'alpha_wtr': alpha_wtr, 'alpha_cmtr': alpha_cmtr, 'alpha_ftr': alpha_ftr, 'alpha_lvtr': alpha_lvtr,
            'beta_ltr': beta_ltr, 'beta_wtr': beta_wtr, 'beta_cmtr': beta_cmtr, 'beta_ftr': beta_ftr, 'beta_lvtr': beta_lvtr}
