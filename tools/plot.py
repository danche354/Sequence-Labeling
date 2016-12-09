'''
plot train and test loss and accuracy
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(train, test, title, x_lable, y_label, folder_path):
    plt.title(title)
    plt.xlabel(x_lable)
    plt.ylabel(y_label)

    plt.plot(train, 'b^-', label='train')
    plt.plot(test, 'r^-', label='test')

    # show grid
    plt.grid(True) 
    plt.legend(loc='upper left')
    plt.savefig(folder_path+'/'+title)
   
