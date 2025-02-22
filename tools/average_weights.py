import sys

import torch

sys.path.append('../')

def load_ckpt(path_to_ckpt):
    ckpt = torch.load(path_to_ckpt, map_location=lambda storage, loc: storage)
    # if hasattr(ckpt, 'state_dict'):
    #     return ckpt['state_dict']
    return dict(ckpt)


params1 = load_ckpt("../output/mappilary/ddrnet23_slim/best_220.pth")
params2 = load_ckpt("../output/mappilary/ddrnet23_slim/revision_3/combo_182+206.pth")


for name1 in params1:
    if name1 in params2:
        params2[name1].data.copy_(0.5 * params1[name1].data + 0.5 * params2[name1].data)


# model.eval()
# model.load_state_dict(params2)
# new_state_dict = {'state_dict': model.state_dict()}
# torch.save(new_state_dict, '../checkpoints/Yolov4_15+26+44.pth')
torch.save(params2, "../output/mappilary/ddrnet23_slim/combo_182+206+220.pth")
#
# sample = torch.ones([1, 3, 64, 64]).to("cuda:0")
# traced = torch.jit.trace(model, torch.rand((1, 3, 256, 1600)))
# traced.save(f"../ckpt/effnetb0_final_stage/traced_effnetb0_averaged.pth")
print("saved")
