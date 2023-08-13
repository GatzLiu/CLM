from params.params_common import *

LR = [0.001, 0.01][dataset]
LAMDA = [0.0, 0.0][dataset]
loss = ['primary', 'click', 'multi-obj', 'unsuper'][3]
exp_weight = {'primary': 0.0, 'click': 1.0, 'multi-obj': 1.0, 'unsuper': 1.0}[loss]
sim_order_weight = {'primary': 0.0, 'click': 0.0, 'multi-obj': 0.0, 'unsuper': 2.0}[loss]
pxtr_reconstruct_weight = {'primary': 0.0, 'click': 0.0, 'multi-obj': 0.0, 'unsuper': 0.1}[loss]
pxtr_weight = [1.0, 1.0, 1.0, 1.0, 1.0]
pxtr_prompt = [2.0, 2.0, 2.0, 2.0, 2.0]
bias_weight = 10.0
layer_num = 5
decay = 0.5
if_debias = [True, False][0]

# model save
all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR, 'LAMDA': LAMDA,
            'PXTR_DIM': PXTR_DIM, 'ITEM_DIM': ITEM_DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': False,
            'TEST_VALIDATION': 'Validation', 'TOP_K': TOP_K,
            'OPTIMIZER': 'Adam', 'SAMPLER': 'MMOE', 'AUX_LOSS_WEIGHT': 0, 'DIR': DIR, 'pxtr_weight': pxtr_weight, 'exp_weight': exp_weight,
            'sim_order_weight': sim_order_weight, 'pxtr_reconstruct_weight': pxtr_reconstruct_weight, 'bias_weight': bias_weight,
            'layer_num': layer_num, 'decay': decay, 'if_debias': if_debias, 'pxtr_prompt': pxtr_prompt}
