from params.params_common import *

trans = 1
transformer = ['transformer', 'set_transformer', 'DC2IN', 'SoGCN', 'orth_transformer'][trans]
LR = [0.02, 0.02, 0.02, 0.02, 0.02][trans]
LR = 0.02

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR,
            'PXTR_DIM': PXTR_DIM, 'ITEM_DIM': ITEM_DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE,
            'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'TOP_K': TOP_K, 'PXTR_LIST': PXTR_LIST, 'OPTIMIZER': 'Adam', 'DIR': DIR,
            'transformer': transformer}