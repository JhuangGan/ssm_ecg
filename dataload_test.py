from argparse import ArgumentParser

from os.path import exists, join, dirname
from ecg_datamodule import ECGDataModule
from dl_models.s4_model import S4Model
import logging
import os
from os.path import exists, join, dirname
import time

def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--dataset",
        dest="target_folder",
        help="used dataset for training",
    )
    parser.add_argument("--logdir", default="./logs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--label_class", default="label_all")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--input_size", type=int, default=250)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--nomemmap", action="store_true", default=False)
    parser.add_argument("--test_folds", nargs="+", default=[9, 10], type=int)
    parser.add_argument("--filter_label")
    parser.add_argument("--combination",  default="both")
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--model", default='xresnet1d50')
    parser.add_argument("--rate", default=1.0, type=float)
    parser.add_argument("--d_state", default=8, type=int)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--s4_dropout", default=0.2, type=float)
    parser.add_argument("--bn", action='store_true', default=False)
    parser.add_argument("--binary_classification",
                        action='store_true', default=False)
    parser.add_argument("--concat_pooling",
                        action='store_true', default=False)
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--use_meta_information_in_head", action='store_true', default=False)
    parser.add_argument("--cpc_bn_encoder", action='store_true', default=False)
    return parser


def init_logger(debug=False, log_dir="./experiment_logs"):
    level = logging.INFO

    if debug:
        level = logging.DEBUG

    # remove all handlers to change basic configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=os.path.join(log_dir, "info.log"),
        level=level,
        format="%(asctime)s %(name)s:%(lineno)s %(levelname)s:  %(message)s  ",
    )
    return logging.getLogger(__name__)

def get_experiment_name(args):
    experiment_name = str(time.asctime()) + "_" + \
        str(time.time_ns())[-3:]

    return experiment_name


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = parse_args(parser)

    args = parser.parse_args()
    print(args)
    experiment_name = get_experiment_name(args)

    init_logger(log_dir=join(args.logdir, experiment_name))

    # data
    datamodule = ECGDataModule(
        args.batch_size,
        args.target_folder,
        label_class=args.label_class,
        num_workers=args.num_workers,
        test_folds=args.test_folds,
        nomemmap=args.nomemmap,
        combination=args.combination,
        filter_label=args.filter_label,
        data_input_size=args.input_size,  ## 需要看一下结果
        normalize=args.normalize,
        use_meta_information_in_head=args.use_meta_information_in_head
    )


