import argparse

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--model_config", type=str, default="vit_configs/cifar_224.gin",
                    help="Where to search for pretrained ViT models.")

parser.add_argument("--name", required=False,
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--dataset", choices=["cifar10", "cifar100", "dtd"], default="cifar10",
                    help="Which downstream task.")
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "So_vit_7",
                                             "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                    default="ViT-B_16",
                    help="Which variant to use.")
parser.add_argument("--pretrained_dir", type=str, default="checkpoints/pretrained_ckpts/ViT-B_16.npz",
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--output_dir", default="checkpoints/output_ckpts", type=str,
                    help="The output directory where checkpoints will be written.")
parser.add_argument("--log_dir", default="checkpoints/logs", type=str,
                    help="The output directory where checkpoints will be written.")

parser.add_argument("--img_size", default=224, type=int,
                    help="Resolution size")
parser.add_argument("--train_batch_size", default=64, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=64, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--eval_every", default=100, type=int,
                    help="Run prediction on validation set every so many steps."
                         "Will always run one evaluation at the end of training.")

parser.add_argument("--learning_rate", default=3e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_steps", default=500, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument("--warmup_steps", default=100, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O2',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
args = parser.parse_args()