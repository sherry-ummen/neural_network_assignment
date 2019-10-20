import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Options:")
    parser.add_argument('-l','--lc', dest='violin_chart',help='Plot Violin chart (Distribution between X and Avg user count)',action='store_true')
    parser.add_argument('-c','--cor', dest='correlation_chart',help='Plot Correlation between Day and Avg user count',action='store_true')
    parser.add_argument('-v','--verbose',dest='verbose', help='Train model with Verbose',action='store_true')
    parser.add_argument('-p','--predict', dest='predict', help='Print the prediction',action='store_true')
    parser.add_argument('-f','--file', dest='file', help='csv file path', default='./test_data/users_elisa.csv')
    return parser.parse_args(args)

def plot_violin(data):
    # Create a figure instance
    fig = plt.figure()
    # Create an axes instance
    ax = fig.add_axes([0.15, 0.1, 0.7, 0.3])
    # Create the violinplot
    vp = ax.violinplot(data.iloc[:,4:].values)
    plt.show(block=False)

def plot_correlation_chart(data):
    new_data = data.drop(data.columns[[0,2,3,4,5,6,7,8,9,10,11,12]], axis=1)
    # plt.clf()
    new_data.groupby('Day').mean().plot(kind='bar', legend=None)
    plt.show(block=False)

def neural_network_mlpregression(data, args):
    np.random.seed(100)
    x_input = data.iloc[:, 1:3].join(data.iloc[:, 5:55]).values
    y_target_output = data.iloc[:, 4].values
    product_model = MLPRegressor(hidden_layer_sizes=(5,),
                              activation='relu',
                              solver='adam',
                              learning_rate='adaptive',
                              max_iter=1500,
                              learning_rate_init=0.001,
                              alpha=0.01, verbose=args.verbose)
    product_model.fit(x_input, y_target_output)
    # Prediction score
    print('Prediction score: %f' % product_model.score(x_input, y_target_output))
    if args.predict:
        print('Prediction')
        print(product_model.predict(x_input))
    # plt.clf()
    pd.DataFrame(product_model.loss_curve_).plot()
    plt.show(block=True)
    print('#'*10,'-Finished-','#'*10)



def main(args):
    args = parse_args(args)
    
    data = pd.read_csv(args.file, header=0, delimiter=';', decimal=',')
 
    # Violin chart
    if(args.violin_chart):
        plot_violin(data)

    # Correlation chart
    if(args.correlation_chart):
        plot_correlation_chart(data)
        
    # Neural Network Training model
    neural_network_mlpregression(data, args)
    

if __name__ == "__main__":
    main(sys.argv[1:])
