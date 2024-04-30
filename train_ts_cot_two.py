
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from ts_cot_model_two import TS_CoT
import tasks
import datautils
from utils import init_cuda, load_config,_logger
import random



def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def eval_mlp(device,logger,args):
    logger.info("="*100)
    logger.info("label ratio :"+args.decomp_mode)
    logger.info("="*100)
    if args.backbone_type == 'TS_CoT':
        
        train_data, train_labels, test_data, test_labels = datautils.get_data_loader(args)
    else:
        logger.info('Unknown Backbone')
        return
    model = TS_CoT(
                input_dims=train_data[0].shape[-1],
                output_dims=args.repr_dims,
                device=device,
                args=args
            )
    model.tem_encoder.load_state_dict(torch.load(args.model_path)['TemEncoder'])
    model.fre_encoder.load_state_dict(torch.load(args.model_path)['FreEncoder'])
    # model.sea_encoder.load_state_dict(torch.load(args.model_path)['SeaEncoder'])
    logger.info('Pre-trained Model Loading...')
    out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels,
                                                eval_protocol=args.eval_protocol, args = args)
    # logger.info(f"Evaluation result: ACC: {eval_res['acc']}   AUROC: {eval_res['auroc']}")
    for evals,val in eval_res.items() :
        logger.info(f"{evals} : {val}")



if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        help='The experimental dataset to be used: HAR, Epi, SleepEDF, Waveform.')
    parser.add_argument('--dataloader', type=str, default=None, help='data loader')
    parser.add_argument('--gpu', type=int, default=0, help='The experimental GPU index.')
    parser.add_argument('--max-threads', type=int, default=8, help='The maximum threads')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--repr-dims', type=int, default=32, help='Dimension of Representation')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs to be trained')
    parser.add_argument('--seed', type=int, default=1024, help='The random seed to be fixed')
    parser.add_argument('--eval', action="store_true", help='Set true for evaluation')
    parser.add_argument('--num-cluster', default='5', type=str, help='number of clusters')
    parser.add_argument('--temperature', default=0.1, type=float, help='softmax temperature of InfoNCE')
    parser.add_argument('--warmup', default=0.50, type=float, help='Warmup epoch before using co-training')
    parser.add_argument('--prototype-lambda', default=0.1, type=float, help='Prototypical loss scale adjustment')
    parser.add_argument('--eval_protocol', default='mlp', type=str, help='Classification backbone for downstreaming tasks.')
    parser.add_argument('--backbone_type', default='TS_CoT', type=str,
                        help='Which backone to use for representation learning. ')
    parser.add_argument('--dropmask', default=0.9, type=float, help='Masking ratio for augmentation')
    parser.add_argument('--model_path', default=None, type=str, help='The path of the model to be loaded')
    parser.add_argument('--ma_gamma', default=0.9999, type=float, help='The moving average parameter for prototype updating')
    parser.add_argument('--run_desc', default="exp_1p", type=str, help='run desc')
    parser.add_argument('--data_perc', default="train", type=str, help='a numer of labeling data')
    parser.add_argument('--decomp_mode', default="seasonal", type=str, help='The mode of to select decomposed mode')
    args = parser.parse_args()

    if args.dataloader is None :
        args.dataloader = args.dataset

    experiment_log_dir = os.path.join("exp_logs", args.run_desc, args.dataset,args.backbone_type)
    os.makedirs(experiment_log_dir, exist_ok=True)
    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
    logger = _logger(log_file_name)

    device = init_cuda(args.gpu, seed=args.seed, max_threads=args.max_threads)
    logger.info('Loading data... ')
    logger.info(args)
    if args.backbone_type == 'TS_CoT':
        # data_perc = args.data_perc
        # args.data_perc = "train"
        # train_data, train_labels, test_data, test_labels = datautils.get_data_loader(args)
        # args.data_perc = data_perc
        train_data, train_labels, test_data, test_labels = datautils.get_data_loader(args)
    else:
        logger.info('Unknown Backbone')
        raise Exception("Unknown Backbone")

    args = load_config(args.dataloader, args)
    args.num_cluster = args.num_cluster.split(',')

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = 'save_dir/test' + '/' + args.dataset + '/' + args.backbone_type+ '/'+ now
    args.run_dir = run_dir
    

    if args.backbone_type == 'TS_CoT' and not args.eval:
        os.makedirs(run_dir, exist_ok=True)
        if args.eval:
            args.epochs = 0
        model = TS_CoT(
            input_dims=train_data[0].shape[-1],
            output_dims=args.repr_dims,
            device=device,
            args=args
        )
        train_model = model.fit_ts_cot(
            train_data,
            n_epochs=args.epochs,
            logger=logger
        )
        model.save(f'{run_dir}/model.pkl')
        logger.info(os.path.basename(__file__))
        args.eval = True 
        args.model_path = f'{run_dir}/model.pkl'
        logger.info(f"save mode path :{args.model_path}")
    else:
        logger.info('Unknown Backbone')
    logger.info(args.eval)
    if args.eval:
        ft_modes=[args.data_perc]
        for ft_mode in ft_modes:
            # args.decomp_mode = ft_mode
            eval_mlp(device=device,logger=logger,args=args)


    logger.info(os.path.basename(__file__))
    logger.info("Finished.")



