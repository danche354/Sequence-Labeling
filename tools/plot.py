'''
plot train and test loss and accuracy
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(train, test, title, x_lable, y1_label, y2_label, folder_path):
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)  
    ax1.plot(train, 'b^-', label='train')
    ax1.legend(loc='upper left')
    ax1.set_title(title)
    ax1.set_xlabel(x_lable)
    ax1.set_ylabel(y1_label)

    ax2 = plt.twinx()
    ax2.plot(test, 'r^-', label='test')
    ax2.legend(loc='upper right')
    ax2.set_ylabel(y2_label)

    # # show grid
    ax1.grid(True) 
    ax2.grid(True) 
    plt.savefig(folder_path+'/'+title)

if __name__ == '__main__':
    plot([1,2,3],[15,10,11],'try','epoch','loss1','loss2','./')
   
