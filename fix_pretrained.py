import re
import torch
import os
from collections import OrderedDict
import model

CHEXNET_PATH = '/content/CheXNet'
N_CLASSES = 14

os.chdir(CHEXNET_PATH)
# original saved file with DataParallel
loaded = torch.load('./pretrained/model.pth.tar')
state_dict = loaded['state_dict']
# create new OrderedDict that does not contain `module.`
# initialize and load the model
model = model.DenseNet121(N_CLASSES).cuda()
model = torch.nn.DataParallel(model).cuda()
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    for layer in ['norm', 'relu', 'conv']:
        if re.search(r'.' + layer + '.[0-9]', k):
            k = k.replace('.' + layer + '.', '.' + layer)
    new_state_dict[k] = v
# load params
model.load_state_dict(new_state_dict)
model.cpu()

print('Now see converted state dict:')
print(new_state_dict.keys())

# saving model:
state = {
    'epoch': loaded['epoch'],
    'arch': loaded['arch'],
    'state_dict': model.state_dict(),
    'optimizer': loaded['optimizer'],
}
torch.save(state, './pretrained/model2.pth')
