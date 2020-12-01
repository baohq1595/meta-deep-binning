import gdown
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--link', help='input google drive download link')
    parser.add_argument('--name', help='input filename to save')
    args = parser.parse_args()
    url = args.link

    output = args.name

    gdown.download(url, output, quiet=False)

