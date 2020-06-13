# directory containing frames
frames_dir = './data/frames'
# directory containing labels and annotations
info_dir = './data/info'

# i3d model pretrained on Kinetics, https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph
i3d_pretrained_path = './data/rgb_i3d_pretrained.pt'

# num of frames in a single video
num_frames = 160

# input data dims;
C, H, W = 3,224,224

# statistics of dataset
label_max = 30
label_min = 6
judge_max = 5
judge_min = 1

# output dimension of I3D backbone
feature_dim = 1024

# For USDL, score is chosen from [6, 7, 8, ..., 30].
# For MUSDL, each judge choose a score from [1, 2, 3, 4, 5].
output_dim = {'USDL':25, 'MUSDL': 5}

# num of judges in MUSDL method
num_judges = 6
