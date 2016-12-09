'''
plot train and test loss and accuracy
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(train, test, title, x_lable, y_label):
    plt.xlabel(x_lable)
    plt.ylabel(y_label)

    plt.plot(train)
    plt.plot(test)
    plt.savefig('')
   

if __name__ == '__main__':
    plot([1,2,3],[2,2,4],title='try',x_lable='epoch',y_label='accuracy')
