import torch
import numpy as np
from tqdm import tqdm
from models.losses import prototype_loss
from dataset.meta_dataset_reader import MetaDatasetEpisodeReader
import argparse
from tabulate import tabulate
import tensorflow as tf
from models.urt import MultiHeadURT, MultiHeadURT_value, get_lambda_urt_avg, apply_urt_avg_selection, get_cosine_logits 
from utils.model_utils import get_domain_extractor, extract_features
import torch.nn.functional as F
from pathlib import Path
from utils.misc_utils import *
import time
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config():

    parser = argparse.ArgumentParser(description='Train URT networks')
    parser.add_argument('--save_dir', type=str, default='checkpoints/output_ckpts', help="The saved path in dir.")
    parser.add_argument('--cache_dir', type=str, help="The saved path in dir.")
    parser.add_argument('--seed', type=int, help="The random seed.")
    parser.add_argument('--interval.train', type=int, default=100, help='The number to log training information')
    parser.add_argument('--interval.test', type=int, default=2000, help='The number to log training information')
    parser.add_argument('--interval.train.reset', type=int, default=500, help='The number to log training information')

    # model args
    parser.add_argument('--model.backbone', default='resnet18', help="Use ResNet18 for experiments (default: False)")
    parser.add_argument('--model.classifier', type=str, default='cosine', choices=['none', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")

    # urt model 
    parser.add_argument('--urt.variant', type=str)
    parser.add_argument('--urt.temp', type=str)
    parser.add_argument('--urt.head', default=2, type=int)
    parser.add_argument('--urt.penalty_coef', default=0.1, type=float)
    # train args
    parser.add_argument('--train.max_iter', type=int, default=10000, help='number of epochs to train (default: 10000)')
    parser.add_argument('--train.weight_decay', type=float, default=1e-5, help="weight decay coef")
    parser.add_argument('--train.optimizer', type=str, default='adam', help='optimization method (default: momentum)')

    parser.add_argument('--train.scheduler', type=str, default='cosine', help='optimization method (default: momentum)')
    parser.add_argument('--train.learning_rate', type=float, default=1e-2, help='learning rate (default: 0.0001)')
    parser.add_argument('--train.lr_decay_step_gamma', type=float, default=0.9, metavar='DECAY_GAMMA')
    parser.add_argument('--train.lr_step', type=int, help='the value to divide learning rate by when decayin lr')

    # Required parameters
    parser.add_argument("--img_size", type=int, default=224,
                            help="Where to search for pretrained ViT models.")
    parser.add_argument("--trainsets", type=str, default="aircraft omniglot",
                        help="Use base ViT model")
    parser.add_argument("--testsets", type=str, default="mnist omniglot traffic_sign",
                        help="Evaluation domains")
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="Where to search for pretrained ViT models.")

    xargs = vars(parser.parse_args())
    return xargs

# def test_all_dataset(args, testsets, test_loader, URT_model, extractors, mode, training_iter, cosine_temp):
def test_all_dataset(args, testsets, test_loader, extractors, URT_model, cosine_temp):
    URT_model.eval()
    our_name   = 'urt'
    accs_names = [our_name]
    alg2data2accuracy = collections.OrderedDict()
    alg2data2accuracy['sur-paper'], alg2data2accuracy['sur-exp'] = pre_load_results()
    alg2data2accuracy[our_name] = {name: [] for name in testsets}

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            our_losses = AverageMeter()
            print(f'Evaluating few shot classification on {dataset} dataset')
            for i in tqdm(range(args['num_tasks'])):
                sample = test_loader.get_test_task(session, dataset)
                context_features = extract_features(extractors, sample['context_images'])
                target_features = extract_features(extractors, sample['target_images'])
                context_labels = sample['context_labels'].to(device)
                target_labels = sample['target_labels'].to(device)
                n_classes = len(np.unique(context_labels.cpu().numpy()))
                # optimize selection parameters and perform feature selection
                avg_urt_params = get_lambda_urt_avg(context_features, context_labels, n_classes, URT_model, normalize=True)
                    
                urt_context_features = apply_urt_avg_selection(context_features, avg_urt_params, normalize=True)
                urt_target_features  = apply_urt_avg_selection(target_features, avg_urt_params, normalize=True) 
                proto_list  = []
                for label in range(n_classes):
                    proto = urt_context_features[context_labels == label].mean(dim=0)
                    proto_list.append(proto)
                urt_proto = torch.stack(proto_list)

                #if random.random() > 0.99:
                #  print("urt avg score {}".format(avg_urt_params))
                #  print("-"*20)
                with torch.no_grad():
                    logits = get_cosine_logits(urt_target_features, urt_proto, cosine_temp)
                    loss   = F.cross_entropy(logits, target_labels)
                    our_losses.update(loss.item())
                    predicts = torch.argmax(logits, dim=-1)
                    final_acc = torch.eq(target_labels, predicts).float().mean().item()
                    alg2data2accuracy[our_name][dataset].append(final_acc)

    show_results(testsets, alg2data2accuracy, ('sur-paper', our_name), print)
    return

def eval_urt_metadataset(args):
  
    # set up logger
    log_dir = Path(args['save_dir']).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args['seed'] is None or args['seed'] < 0:
        seed = len(list(Path(log_dir).glob("*.txt")))
    else:
        seed = args['seed']

    img_size = args['img_size']
    trainsets = args['trainsets'].split(' ')
    testsets = args['testsets'].split(' ')

    config_file = f'dataset/configs/meta_dataset_{img_size}x{img_size}.gin'
    print(trainsets)
    extractors=dict()
    extractors['imagenet'] = get_domain_extractor(img_size, domain='imagenet').to(device)
    extractors['imagenet'].eval()
    # for trainset in trainsets:
    #     extractors[trainset] = get_domain_extractor(img_size, domain=trainset).to(device)
    #     extractors[trainset].eval()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    train_loader = MetaDatasetEpisodeReader('train', trainsets, testsets, testsets, config_file=config_file)
    val_loader = MetaDatasetEpisodeReader('val', trainsets, trainsets, testsets, config_file=config_file)
    test_loader = MetaDatasetEpisodeReader('test', trainsets, testsets, testsets, config_file=config_file)

    # init prop model
    URT_model  = MultiHeadURT(key_dim=256, query_dim=len(extractors)*256, hid_dim=512, temp=1, att="dotproduct", n_head=args['urt.head'])
    URT_model  = torch.nn.DataParallel(URT_model)
    URT_model  = URT_model.to(device)
    cosine_temp = torch.nn.Parameter(torch.tensor(10.0).to(device))
    params = [p for p in URT_model.parameters()] + [cosine_temp]

    optimizer  = torch.optim.Adam(params, lr=args['train.learning_rate'], weight_decay=args['train.weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args['train.max_iter'])

    # load checkpoint optional
    last_ckp_path = log_dir / 'last-ckp-seed-{:}.pth'.format(seed)
    if last_ckp_path.exists():
        checkpoint  = torch.load(last_ckp_path)
        start_iter  = checkpoint['train_iter'] + 1
        URT_model.load_state_dict(checkpoint['URT_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        print ('load checkpoint from {:}'.format(last_ckp_path))
    else:
        print ('randomly initialiization')
        start_iter = 0
    max_iter = args['train.max_iter']

    our_losses, our_accuracies = AverageMeter(), AverageMeter()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        if 1: #for dataset in trainsets:
            for index in tqdm(range(max_iter)):

                sample = train_loader.get_train_task(session)
                context_features = extract_features(extractors, sample['context_images'].to(device))
                target_features = extract_features(extractors, sample['target_images'].to(device))
                context_labels = sample['context_labels'].to(device)
                target_labels = sample['target_labels'].to(device)

                URT_model.train()
                n_classes = len(np.unique(context_labels.cpu().numpy()))
                # optimize selection parameters and perform feature selection
                avg_urt_params = get_lambda_urt_avg(context_features, context_labels, n_classes, URT_model, normalize=True)
                # identity matrix panelize to be sparse, only focus on one aspect
                penalty = torch.pow( torch.norm( torch.transpose(avg_urt_params, 0, 1) @ avg_urt_params - torch.eye(args['urt.head']).to(device) ), 2)
                # n_samples * (n_head * 512)
                urt_context_features = apply_urt_avg_selection(context_features, avg_urt_params, normalize=True)
                urt_target_features  = apply_urt_avg_selection(target_features, avg_urt_params, normalize=True) 
                proto_list  = []
                for label in range(n_classes):
                    proto = urt_context_features[context_labels == label].mean(dim=0)
                    proto_list.append(proto)
                urt_proto = torch.stack(proto_list)
                logits = get_cosine_logits(urt_target_features, urt_proto, cosine_temp) 
                loss = F.cross_entropy(logits, target_labels) + args['urt.penalty_coef']*penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                with torch.no_grad():
                    predicts  = torch.argmax(logits, dim=-1)
                    final_acc = torch.eq(target_labels, predicts).float().mean().item()
                    our_losses.update(loss.item())
                    our_accuracies.update(final_acc * 100)

                print("{:} [{:5d}/{:5d}] [OUR] lr: {:}, loss: {:.5f}, accuracy: {:.4f}".format(time_string(), index, max_iter, lr_scheduler.get_last_lr(), our_losses.avg, our_accuracies.avg))

                if (index+1) % args['interval.test'] == 0 or index+1 == max_iter:
                    test_all_dataset(args, trainsets, val_loader, extractors, URT_model, cosine_temp)
                    test_all_dataset(args, testsets, test_loader, extractors, URT_model, cosine_temp)
    return


def main():
    args = load_config()
    print(args)
    eval_urt_metadataset(args)
    return

if __name__ == '__main__':
  main()