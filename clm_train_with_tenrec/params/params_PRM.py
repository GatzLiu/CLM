from params.params_common import *

LR = [0.001, 0.015][dataset]
LAMDA = [0.0, 0.0][dataset]
pxtr_weight = [1.0, 1.0, 1.0]
exp_weight = 1.0
sim_order_weight = 0.0 # 2.0
pxtr_reconstruct_weight = 0.0 # 0.1
# model save

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR, 'LAMDA': LAMDA,
            'PXTR_DIM': PXTR_DIM, 'ITEM_DIM': ITEM_DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': False,
            'TEST_VALIDATION': 'Validation', 'TOP_K': TOP_K, 'PXTR_LIST': PXTR_LIST,
            'OPTIMIZER': 'Adam', 'SAMPLER': 'MMOE', 'AUX_LOSS_WEIGHT': 0, 'DIR': DIR, 'pxtr_weight': pxtr_weight, 'exp_weight': exp_weight,
            'sim_order_weight': sim_order_weight, 'pxtr_reconstruct_weight': pxtr_reconstruct_weight}