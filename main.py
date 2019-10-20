import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt



def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Options:")
    parser.add_argument('-l','--lc', dest='violin_chart',help='Plot Violin chart (Distribution between X and Avg user count)',action='store_true')
    parser.add_argument('-c','--cor', dest='correlation',help='Plot Correlation between Day and Avg user count')
    parser.add_argument('-v','--verbose',dest='verbose', help='Train model with Verbose')
    parser.add_argument('-p','--predict', dest='predict', help='Print the prediction')
    parser.add_argument('-f','--file', dest='file', help='csv file path', default='./test_data/users_elisa.csv')
    return parser.parse_args(args)

def plot_violin(data):
    # Create a figure instance
    fig = plt.figure()
    # Create an axes instance
    ax = fig.add_axes([0.15, 0.1, 0.7, 0.3])
    # Create the violinplot
    vp = ax.violinplot(data.iloc[:,4:].values)
    plt.show()


def main(args):
    args = parse_args(args)
    
    data = pd.read_csv(args.file, header=0, delimiter=';', decimal=',')
 
    # Violin chart
    if(args.violin_chart):
        plot_violin(data)




if __name__ == "__main__":
    main(sys.argv[1:])
