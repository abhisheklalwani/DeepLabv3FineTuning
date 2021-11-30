from os import P_DETACH
import pandas as pd
import click
import matplotlib.pyplot as plt

@click.command()
@click.option("--exp_path",
              required=True,
              help="Experiments for which the visualization are to be generated")

def generate_visualization(exp_path):

    df = pd.read_csv(exp_path+'/log.csv')
    print(df)
	#df.plot(kind='line',x='epoch',y='Train_loss')
	#plt.savefig('results/WeightedCrossEntropy25Epochs/SegmentationOutput.png',bbox_inches='tight')
    plt.plot(df['epoch'], df['Train_loss'], label='Training Loss')
    plt.plot(df['epoch'], df['Validation_loss'], label='Validation Loss')
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Training and Validation Loss')
    plt.legend(loc="upper right")
    plt.savefig(exp_path+'/Loss.png',bbox_inches='tight')
    

generate_visualization()