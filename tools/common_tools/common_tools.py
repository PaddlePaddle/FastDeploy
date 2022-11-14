import argparse


def argsparser():

    parser = argparse.ArgumentParser(description=__doc__)
    ## argumentments for auto compression
    parser.add_argument('--auto_compress', default=False, action='store_true')
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help="choose PTQ or QAT as quantization method",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")

    ## arguments for other tools
    return parser


def main():

    args = argsparser().parse_args()
    if args.auto_compress == True:
        try:
            from fd_auto_compress.fd_auto_compress import auto_compress
            print("Welcome to use FastDeploy Auto Compression Toolkit!")
            auto_compress(args)
        except ImportError:
            print(
                "Can not start auto compresssion successfully! Please check if you have installed it!"
            )


if __name__ == '__main__':
    main()
