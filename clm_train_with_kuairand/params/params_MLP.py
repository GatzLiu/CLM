from params.params_common import *

LR = [0.001, 0.01][dataset]
LAMDA = [0.0, 0.0][dataset]
loss = ['primary', 'click', 'multi-obj', 'unsuper'][3]
exp_weight = {'primary': 0.0, 'click': 1.0, 'multi-obj': 0.0, 'unsuper': 1.0}[loss]
sim_order_weight = {'primary': 0.0, 'click': 0.0, 'multi-obj': 0.0, 'unsuper': 2.0}[loss]
pxtr_reconstruct_weight = {'primary': 0.0, 'click': 0.0, 'multi-obj': 0.0, 'unsuper': 0.1}[loss]
primary_weight = {'primary': 1.0, 'click': 0.0, 'multi-obj': 0.0, 'unsuper': 0.0}[loss]
multi_object_weight = {'primary': 0.0, 'click': 0.0, 'multi-obj': 1.0, 'unsuper': 0.0}[loss]
pxtr_weight_for_ranking_sim_loss = [1.0, 1.0, 1.0, 1.0, 1.0]
pxtr_weight_for_multi_object = [0.2, 0.3, 0.8, 3.0, 0.0]
mode = ['LR', 'MLP'][1]
# model save

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR, 'LAMDA': LAMDA,
            'PXTR_DIM': PXTR_DIM, 'ITEM_DIM': ITEM_DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': False,
            'TEST_VALIDATION': 'Validation', 'TOP_K': TOP_K,
            'OPTIMIZER': 'Adam', 'SAMPLER': 'MMOE', 'AUX_LOSS_WEIGHT': 0, 'DIR': DIR, 'pxtr_weight': pxtr_weight_for_ranking_sim_loss, 'exp_weight': exp_weight,
            'sim_order_weight': sim_order_weight, 'pxtr_reconstruct_weight': pxtr_reconstruct_weight, 'primary_weight': primary_weight,
            'multi_object_weight': multi_object_weight, "mode": mode, 'pxtr_prompt': pxtr_weight_for_multi_object}