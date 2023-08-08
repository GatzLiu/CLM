import json
import random as rd

path_train_read = "train_data_bp.json"
path_validate = "validation_data.json"
path_train = "train_data.json"

def write_data(data, path):
    f = open(path, 'w')
    jsObj = json.dumps(data)
    f.write(jsObj)
    f.close()

with open(path_train_read) as f:
    line = f.readline()
    data = json.loads(line)
    f.close()
print('before shuffle', data[0: 5])
rd.shuffle(data)
print('after shuffle', data[0: 5])
samp_num = len(data)
train_data = data[0: int(0.9 * samp_num)]
validation_data = data[int(0.9 * samp_num):]
write_data(train_data, path_train)
write_data(validation_data, path_validate)
