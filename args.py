import argparse


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        default="False",
        help="bool value to indicate wether to use pretrained weights for the backbone network",
    )
    parser.add_argument(
        "--dataset_path",
        help='path to trainset with category name, eg: "../data/foregeynet/',
    )
    parser.add_argument(
        "--augmented_dataset_path", help="path to offline-generated augmentations"
    )
    parser.add_argument(
        "--data_augmentation_type",
        default="none",
        help="options: ['none', 'basic']",
    )
    parser.add_argument(
        "--dims",
        default=[512, 512, 512, 512, 512, 512, 512, 512, 128],
        help="list indicating number of hidden units for each layer of projection head",
    )
    parser.add_argument("--num_class", default=6)
    parser.add_argument(
        "--manipulation_type",
        default="6transforms",
        help="Type of manipulation: 6 types of transformations can be applied.",
    )
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument(
        "--freeze_layers",
        default="False",
        help="when true, freezes layer until 2nd layer",
    )
    parser.add_argument("--learning_rate", default=0.03, type=float)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--weight_decay", default=0.0003)
    parser.add_argument("--num_epochs", default=300)
    parser.add_argument("--num_gpus", default=1)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--log_dir", default=r"tb_logs")
    parser.add_argument("--log_dir_name", default=r"exp1")
    parser.add_argument("--checkpoint_filename", default=r"weights")
    parser.add_argument("--monitor_checkpoint", default=r"train_loss")
    parser.add_argument("--monitor_checkpoint_mode", default=r"min")

    args = parser.parse_args()
    return args
