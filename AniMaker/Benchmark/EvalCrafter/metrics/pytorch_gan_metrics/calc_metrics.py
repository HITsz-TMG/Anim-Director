"""Calculate FID and Inception Score of images in a directory."""

import argparse

from torch.utils.data import DataLoader

from . import ImageDataset, get_inception_score_and_fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score and FID")
    parser.add_argument('--path', type=str, required=True,
                        help='path to image directory')
    parser.add_argument('--stats', type=str, required=True,
                        help='precalculated reference statistics')
    parser.add_argument('--use_torch', action='store_true',
                        help='using pytorch as the matrix operations backend')
    parser.add_argument("--num_images", type=int, default=None,
                        help="the number of images to calculate FID")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="dataloader workers")
    args = parser.parse_args()

    dataset = ImageDataset(root=args.path, num_images=args.num_images)
    loader = DataLoader(dataset, batch_size=50, num_workers=args.num_workers)
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader,
        args.stats,
        use_torch=args.use_torch,
        verbose=True)
    print(IS, IS_std, FID)
