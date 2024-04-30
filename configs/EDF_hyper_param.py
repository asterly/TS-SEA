class Param(object):
    def __init__(self):
        # model configs
        self.batch_size = 512
        self.lr = 0.001
        self.repr_dims = 1024
        self.epochs = 20
        self.num_cluster = '5,10'
        self.backbone_type = 'TS_SEA'
        # self.model_path = "pretrained_model/EDF_model.pkl"
 

