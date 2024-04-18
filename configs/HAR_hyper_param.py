class Param(object):
    def __init__(self):
        # model configs
        self.batch_size = 128
        self.lr = 0.001
        self.repr_dims = 512
        self.final_out_channels = 256
        self.epochs = 40
        self.num_cluster = '6'
        #self.backbone_type = 'TS_CoT'
        # self.model_path = "pretrained_model/HAR_model.pkl"
        self.features_len = 18
        self.dropout = 0.35
        self.TC = TC()
        self.Context_Cont = Context_Cont_configs()

class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True