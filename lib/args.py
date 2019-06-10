import argparse


def get_args():
    parser = get_common_parser()
    args = parser.parse_args()
    return args


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', default='data')
    parser.add_argument('-o', '--out-dir', default='out')
    parser.add_argument('-n', '--n-epochs', default=3, type=int)
    parser.add_argument('--batch-size-train', default=64, type=str)
    parser.add_argument('--batch-size-test', default=1000, type=str)
    parser.add_argument('--learning-rate', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--log-interval', default=10, type=int)
    return parser
