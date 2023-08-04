from params.params_common import MODEL
if MODEL == "CLM": from params.params_CLM import all_para
# if MODEL == "NCF": from params.params_NCF import all_para
# if MODEL == "NGCF": from params.params_NGCF import all_para
# if MODEL == "LightGCN": from params.params_LightGCN import all_para
# if MODEL == "LGCN": from params.params_LGCN import all_para

# from params import all_para
from train_model import *
from utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = all_para['GPU_INDEX']

if __name__ == '__main__':
    ## print model hyperparameters
    print_params(all_para)
    ## train the model
    train_model(all_para)

