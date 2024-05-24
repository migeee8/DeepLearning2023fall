import matplotlib.pyplot as plt
from cnn import *
from data_utils import get_CIFAR10_data
from solver import Solver

data = get_CIFAR10_data()
model = ThreeLayerCovNet(reg=0.9)
solver = Solver(model, data,
                lr_decay=0.05,
                print_every=10,
                num_epochs=5,
                batch_size=2,
                update_rule='sgd_momentum',
                optim_config={'learning_rate': 5e-4, 'momentum': 0.9})

solver.train()

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')
plt.show()

plt.subplot(2, 1, 1)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
