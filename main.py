import sys
import argparse


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Options:")
    parser.add_argument('-l','--lc', dest='violin_chart',help='Plot Violin chart (Distribution between X and Avg user count)')
    parser.add_argument('-c','--cor', dest='correlation',help='Plot Correlation between Day and Avg user count')
    parser.add_argument('-v','--verbose',dest='verbose', help='Train model with Verbose')
    parser.add_argument('-p','--predict', dest='predict', help='Print the prediction')
    parser.add_argument('-f','--file', dest='file', help='csv file path', default='./test_data/users_elisa.csv')
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    print(args.file)

if __name__ == "__main__":
    main(sys.argv[1:])
