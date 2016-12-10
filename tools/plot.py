'''
plot train and test loss and accuracy
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_loss(train, test, folder_path, title, x_lable='epoch', y1_label='train loss', y2_label='dev loss'):
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)  
    ax1.plot(train, 'b^-', label=y1_label)
    ax1.legend(loc='upper left')
    ax1.set_title(title+' loss')
    ax1.set_xlabel(x_lable)
    ax1.set_ylabel(y1_label)

    ax2 = plt.twinx()
    ax2.plot(test, 'r^-', label=y2_label)
    ax2.legend(loc='upper right')
    ax2.set_ylabel(y2_label)

    # # show grid
    ax1.grid(True) 
    ax2.grid(True) 
    plt.savefig(folder_path+'/'+title+' loss')

def plot_accuracy(accuracy, folder_path, title, x_lable='epoch', y_label='dev accuracy'):
    fig = plt.figure()  
    ax = fig.add_subplot(111) 
    ax.plot(accuracy, 'b^-', label='dev accuracy')
    ax.legend(loc='upper left')
    ax.set_title(title+' dev accuracy')
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)

    ax.grid(True)

    plt.savefig(folder_path+'/'+title+' dev accuracy')

if __name__ == '__main__':
    plot_accuracy([1,2,3],'./', title='dd')
   
