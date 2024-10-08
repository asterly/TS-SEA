from torch.utils.data import TensorDataset, DataLoader
from models import *

from utils import *
from tqdm import tqdm
import torch.nn as nn
from info_nce import *

def generate_binomial_mask(target_matrix, p=0.5, seed_=42):
    # np.random.seed(seed_)
    return torch.from_numpy(np.random.binomial(1, p, size=(target_matrix.shape)))

class TS_CoT(nn.Module):
    '''The Proposed TS_CoT model'''

    def __init__(
            self,
            input_dims,
            output_dims=32,
            hidden_dims=64,
            device='cuda',
            lr=0.001,
            args=None
    ):

        super().__init__()
        self.device = device
        self.lr = lr
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        if args.dataset == 'SleepEDF':
            self.tem_encoder = Conv_Pyram_model_EDF(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_EDF(input_dims=input_dims, output_dims=output_dims).to(self.device)
        elif args.dataset == 'HAR' or args.dataset == 'UEA':
            self.tem_encoder = Conv_Pyram_model_HAR(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_HAR(input_dims=input_dims, output_dims=output_dims).to(self.device)
            
        elif args.dataset == 'Epilepsy':
            self.tem_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
        elif args.dataset == 'Waveform':
            self.tem_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
        elif args.dataset == 'ISRUC':
            self.tem_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)
        elif args.dataset == 'RoadBank' or args.dataset == "Bridge":
            self.tem_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)

        else:
            print('Unknown Dataset')
        # self.tc = TC(args, device).to(self.device)

        self.n_epochs = 0
        self.args = args
        self.cluster_result = None
        self.batch_size = args.batch_size


    def fit_ts_cot(self, train_data, train_labels,n_epochs=None,logger=None):

        train_dataset1 = TwoViewloader(train_data)
        train_loader = DataLoader(train_dataset1, batch_size=min(self.batch_size, len(train_dataset1)), shuffle=True,
                                  drop_last=True)

        train_labels = torch.LongTensor(train_labels).to(self.device)
        params = list(list(self.tem_encoder.parameters()) + list(self.fre_encoder.parameters()))
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        cluster_result = None
        last_clusters = None

        self.args.warmup = int(self.args.warmup * self.args.epochs)
        while True:
            # if self.n_epochs>0:
            #     print('Training Epoch '+str(self.n_epochs))
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            batch_loss_tem_nce = [0.0]
            batch_loss_fre_nce = [0.0]
            batch_loss_proto_tem = [0.0]
            batch_loss_proto_fre = [0.0]

            if self.n_epochs == max(int(self.args.warmup), 0):
                features = self.encode(train_data )
                feature_split = int(features.shape[1]/2)
                features_tem = features[:, :feature_split]
                features_fre = features[:, feature_split:]

                cluster_result_tem =  {'im2cluster': [], 'centroids': [], 'density': [], 'ma_centroids':[]}
                cluster_result_fre = {'im2cluster': [], 'centroids': [], 'density': [],  'ma_centroids':[]}
                tem_last_clusters = {'im2cluster': [], 'centroids': [], 'density': [],  'ma_centroids':[]}
                fre_last_clusters = {'im2cluster': [], 'centroids': [], 'density': [],  'ma_centroids':[]}


                #last_clusters = []
                for num_cluster in self.args.num_cluster:
                    cluster_result_tem['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                    cluster_result_fre['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                    # cluster_result_tem['im2cluster'].append(train_labels)
                    # cluster_result_fre['im2cluster'].append(train_labels)
                    cluster_result_tem['centroids'].append(
                        torch.zeros(int(num_cluster), train_data[0].shape[1] * self.args.repr_dims).cuda())
                    cluster_result_fre['centroids'].append(
                        torch.zeros(int(num_cluster), train_data[0].shape[1] * self.args.repr_dims).cuda())
                    cluster_result_tem['density'].append(torch.zeros(int(num_cluster)).cuda())
                    cluster_result_fre['density'].append(torch.zeros(int(num_cluster)).cuda())
                    
                    tem_last_clusters['centroids'].append(torch.zeros(int(num_cluster),  self.args.repr_dims).cuda())
                    fre_last_clusters['centroids'].append(torch.zeros(int(num_cluster),  self.args.repr_dims).cuda())
                    for tmpp in range(int(num_cluster)):
                        lab_idx = torch.nonzero(torch.eq(train_labels,tmpp)).cpu()
                        semi_fre_cenroid = torch.mean(
                            torch.tensor(features_fre[lab_idx]),
                            0).cuda() / torch.norm(torch.mean(torch.tensor(features_fre[lab_idx]),
                            0).cuda()
                        )
                        semi_tem_cenroid = torch.mean(
                            torch.tensor(features_tem[lab_idx]),
                            0).cuda() / torch.norm(torch.mean(torch.tensor(features_tem[lab_idx]),
                            0).cuda()
                        )
                        fre_last_clusters['centroids'][0][tmpp]=semi_fre_cenroid
                        tem_last_clusters['centroids'][0][tmpp]=semi_tem_cenroid
                    cluster_result_tem = run_kmeans(features_tem, self.args, tem_last_clusters)
                    cluster_result_fre = run_kmeans(features_fre, self.args, fre_last_clusters)

                for tmp in range(len(self.args.num_cluster)):
                    
                    

                    tem_im2cluster = cluster_result_tem['im2cluster'][tmp]
                    fre_im2cluster = cluster_result_fre['im2cluster'][tmp]
                   
                    # tem_im2cluster = train_labels 
                    # fre_im2cluster = train_labels 
                    dist_tem = cluster_result_tem['distance_2_center'][tmp]
                    dist_fre = cluster_result_fre['distance_2_center'][tmp]
                    print("=="*50)
                    tmp_cluster = self.args.num_cluster[tmp]
                    for tmpp in range(int(tmp_cluster)):
                        
                        sort_tem_index = np.array(np.argsort(dist_tem[:,tmpp])) #
                        sort_fre_index = np.array(np.argsort(dist_fre[:,tmpp])) #
                        sort_tem_index = sort_tem_index[:int(0.8*len(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu())))]
                        sort_fre_index = sort_fre_index[:int(0.8*len(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu())))]

                        set_tem = np.intersect1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        set_fre = np.intersect1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        neg_tem = np.setdiff1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        neg_fre = np.setdiff1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        if len(neg_tem) > 0 and len(set_tem) > 0:
                            tem_im2cluster[neg_tem] = -1
                        if len(neg_fre) > 0 and len(set_fre) > 0:
                            fre_im2cluster[neg_fre] = -1
                        
                        # cluster_result_fre['centroids'][tmp][tmpp, :] = torch.mean(
                        #     torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),
                        #     0).cuda() / torch.norm(torch.mean(
                        #     torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),
                        #     0).cuda())
                        # cluster_result_tem['centroids'][tmp][tmpp, :] = torch.mean(
                        #     torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]),
                        #     0).cuda() / torch.norm(torch.mean(
                        #     torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]),
                        #     0).cuda())
                        lab_idx = torch.nonzero(torch.eq(train_labels,tmpp)).cpu()
                        semi_fre_cenroid = torch.mean(
                            torch.tensor(features_fre[lab_idx]),
                            0).cuda() / torch.norm(torch.mean(torch.tensor(features_fre[lab_idx]),
                            0).cuda()
                        )
                        semi_tem_cenroid = torch.mean(
                            torch.tensor(features_tem[lab_idx]),
                            0).cuda() / torch.norm(torch.mean(torch.tensor(features_tem[lab_idx]),
                            0).cuda()
                        )
                        cluster_result_fre['centroids'][tmp][tmpp, :] = semi_fre_cenroid
                        cluster_result_tem['centroids'][tmp][tmpp, :] = semi_tem_cenroid


                cluster_result_tem['ma_centroids'] = cluster_result_tem['centroids']
                cluster_result_fre['ma_centroids'] = cluster_result_fre['centroids']
                print(cluster_result_fre['centroids'][0].shape)

            if self.n_epochs > max(int(self.args.warmup), 0):
                features = self.encode(train_data )
                feature_split = int(features.shape[1]/2)
                features_tem = features[:, :feature_split]
                features_fre = features[:, feature_split:]
                for jj in range(len(self.args.num_cluster)):
                    ma_centroids_tem = cluster_result_tem['ma_centroids'][jj]/torch.norm(cluster_result_tem['ma_centroids'][jj], dim=1, keepdim=True)
                    cp_tem = torch.matmul(torch.tensor(features_tem).cuda(), ma_centroids_tem.transpose(1, 0))
                    cluster_result_tem['im2cluster'][jj] = torch.argmax(cp_tem, dim=1)
                    cluster_result_tem['distance_2_center'][jj] = 1-cp_tem.cpu().numpy()
                    ma_centroids_fre = cluster_result_fre['ma_centroids'][jj]/torch.norm(cluster_result_fre['ma_centroids'][jj], dim=1, keepdim=True)
                    cp_fre = torch.matmul(torch.tensor(features_fre).cuda(), ma_centroids_fre.transpose(1, 0))
                    cluster_result_fre['im2cluster'][jj] = torch.argmax(cp_fre, dim=1)
                    cluster_result_fre['distance_2_center'][jj] = 1-cp_fre.cpu().numpy()
                    cluster_result_tem['density'][jj] = torch.ones(cluster_result_tem['density'][jj].shape).cuda()
                #     cluster_result_fre['density'][jj] = torch.ones(cluster_result_fre['density'][jj].shape).cuda()
                
                # if np.isnan(features_tem).any() :
                #     print(features_tem)
                # if np.isnan(features_fre).any() :
                #     print(features_fre)
                # cluster_result_tem = run_kmeans(features_tem, self.args, tem_last_clusters)
                # cluster_result_fre = run_kmeans(features_fre, self.args, fre_last_clusters)
                for tmp in range(len(self.args.num_cluster)):
                    
                    # tem_im2cluster = train_labels 
                    # fre_im2cluster = train_labels 
                    # print(cluster_result_tem['im2cluster'][tmp])
                    tem_im2cluster = cluster_result_tem['im2cluster'][tmp]
                    fre_im2cluster = cluster_result_fre['im2cluster'][tmp]
                    dist_tem = cluster_result_tem['distance_2_center'][tmp]
                    dist_fre = cluster_result_fre['distance_2_center'][tmp]
                    tmp_cluster = self.args.num_cluster[tmp]
                    for tmpp in range(int(tmp_cluster)):
                        keep_ratio = 1
                        sort_tem_index = np.array(np.argsort(dist_tem[:,tmpp])) # 前面的距离小
                        sort_fre_index = np.array(np.argsort(dist_fre[:,tmpp])) #
                        sort_tem_index = sort_tem_index[:int(keep_ratio*len(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu())))]
                        sort_fre_index = sort_fre_index[:int(keep_ratio*len(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu())))]
                        set_tem = np.intersect1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        set_fre = np.intersect1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        neg_tem = np.setdiff1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        neg_fre = np.setdiff1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)

                        if len(neg_tem) > 0 and len(set_tem) > 0:
                            #print(fre_im2cluster,neg_tem)
                            tem_im2cluster[neg_tem] = -1
                        if len(neg_fre) > 0 and len(set_fre) > 0:
                            fre_im2cluster[neg_fre] = -1
                        
                        cluster_result_fre['centroids'][tmp][tmpp, :] = torch.mean(
                            torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda() / torch.norm(torch.mean(
                            torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda())
                        cluster_result_tem['centroids'][tmp][tmpp, :] = torch.mean(
                            torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda() / torch.norm(torch.mean(
                            torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda())
                        # lab_idx = torch.nonzero(torch.eq(train_labels,tmpp)).cpu()
                        # semi_fre_cenroid = torch.mean(
                        #     torch.tensor(features_fre[lab_idx]),
                        #     0).cuda() / torch.norm(torch.mean(torch.tensor(features_fre[lab_idx]),
                        #     0).cuda()
                        # )
                        # semi_tem_cenroid = torch.mean(
                        #     torch.tensor(features_tem[lab_idx]),
                        #     0).cuda() / torch.norm(torch.mean(torch.tensor(features_tem[lab_idx]),
                        #     0).cuda()
                        # )
                        # cluster_result_fre['centroids'][tmp][tmpp, :] = semi_fre_cenroid
                        # cluster_result_tem['centroids'][tmp][tmpp, :] = semi_tem_cenroid
                        if torch.isnan(cluster_result_fre['centroids'][tmp][tmpp, :]).any():
                            # print(cluster_result_fre['centroids'][tmp][tmpp, :])
                            # print(tem_im2cluster , tmpp)
                            print("find none")
                cluster_result_tem['ma_centroids'] = cluster_result_tem['centroids']
                cluster_result_fre['ma_centroids'] = cluster_result_fre['centroids']

            tq_bar = tqdm(train_loader) 
            # nt_xent_criterion = NTXentLoss(self.device, self.args.batch_size, self.args.Context_Cont.temperature,
            #                                self.args.Context_Cont.use_cosine_similarity)
            lambda1 = 1
            lambda2 = 0.7
            for indexs, sample_tem, sample_fre in tq_bar:

                optimizer.zero_grad()
                sample_tem = sample_tem.to(self.device)
                sample_fre = sample_fre.to(self.device)
                tem_feat_1, tem_feat_2, tem_feat_3, tem_z = self.tem_encoder(sample_tem.float(), 1)
                tem_feat_1_m, tem_feat_2_m, tem_feat_3_m, tem_z_m = self.tem_encoder(sample_tem.float(), self.args.dropmask, self.args.seed)
                # print("tem_feat_3 shape",tem_feat_3.shape)
                # print("tem_feat_3_m shape",tem_feat_3_m.shape)
                
                # temp_cont_loss1, temp_cont_feat1 = self.tc(tem_feat_3,tem_feat_3_m)
                # temp_cont_loss2, temp_cont_feat2 = self.tc(tem_feat_3_m,tem_feat_3)

                
                # print("temp_cont_feat1 shape",temp_cont_feat1.shape)
                # print("temp_cont_feat2 shape",temp_cont_feat2.shape)
                # temp_cont_feat1 = F.normalize(temp_cont_feat1, dim=1)
                # temp_cont_feat2 = F.normalize(temp_cont_feat2, dim=1)
                # temp_loss = nt_xent_criterion(temp_cont_feat1, temp_cont_feat2)
                
                fre_feat_1, fre_feat_2, fre_feat_3, fre_z = self.fre_encoder(sample_fre.float(), 1)
                fre_feat_1_m, fre_feat_2_m, fre_feat_3_m, fre_z_m = self.fre_encoder(sample_fre.float(), self.args.dropmask, self.args.seed)

                # temp_cont_feat1 = F.normalize(temp_cont_feat1, dim=1)
                # temp_cont_feat2 = F.normalize(temp_cont_feat2, dim=1)
                # fre_cont_loss1, fre_cont_feat1 = self.tc(fre_feat_3,fre_feat_3_m)
                # fre_cont_loss2, fre_cont_feat2 = self.tc(fre_feat_3_m,fre_feat_3)
                # fre_loss = nt_xent_criterion(fre_cont_feat1, fre_cont_feat2)

                criterion = InfoNCE(self.args.temperature)
                loss_tem_nce = criterion(tem_z.squeeze(-1), tem_z_m.squeeze(-1))
                batch_loss_tem_nce.append(loss_tem_nce.item())
                loss_fre_nce = criterion(fre_z.squeeze(-1), fre_z_m.squeeze(-1))
                batch_loss_fre_nce.append(loss_fre_nce.item())
                loss = loss_fre_nce + loss_tem_nce 
                # loss =(temp_cont_loss1+temp_cont_loss2)* lambda1  + loss_fre_nce + loss_tem_nce
                # loss = loss + (temp_loss)* lambda2
                if self.n_epochs > (self.args.warmup):
                    loss_prototype_tem, cluster_result_tem['ma_centroids']= prototype_loss_cotrain(tem_z , indexs, cluster_result_tem, self.args)
                    loss_prototype_fre, cluster_result_fre['ma_centroids'] = prototype_loss_cotrain(fre_z, indexs, cluster_result_fre, self.args)
                    batch_loss_proto_tem.append(loss_prototype_tem.item())
                    batch_loss_proto_fre.append(loss_prototype_fre.item())
                    loss += loss_prototype_tem * self.args.prototype_lambda
                    loss += loss_prototype_fre * self.args.prototype_lambda
                    if torch.isnan(loss).any():
                        continue
                tq_bar.set_description(f"epoch: {self.n_epochs}/{self.args.epochs} | loss : {loss.item()}")
                loss.backward()
                optimizer.step()
            if logger :
                logger.info(f"epoch: {self.n_epochs}/{self.args.epochs} | loss : {loss.item()}")
            self.n_epochs += 1
        self.cluster_result = cluster_result
        return 0


    def encode(self, data, batch_size=None):
        self.tem_encoder.eval()
        self.fre_encoder.eval()

        train_dataset1 = TwoViewloader(data)
        loader = DataLoader(train_dataset1, batch_size=min(self.batch_size, len(train_dataset1)), shuffle=False,
                                  drop_last=False)

        with torch.no_grad():
            output = []
            for indexs, sample_tem, sample_fre in loader:
                sample_tem = sample_tem.to(self.device)
                sample_fre = sample_fre.to(self.device)
                _, _, _, tem_z = self.tem_encoder(sample_tem.float())
                _, _, _, fre_z = self.fre_encoder(sample_fre.float())

                out = torch.cat((tem_z.squeeze(-1), fre_z.squeeze(-1)), dim=1)

                output.append(out)

            output = torch.cat(output, dim=0)

        return output.cpu().numpy()



    def save(self, fn):
        torch.save({'TemEncoder': self.tem_encoder.state_dict(), 'FreEncoder': self.fre_encoder.state_dict()}, fn)
