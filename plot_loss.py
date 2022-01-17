from matplotlib import pyplot as plt
from progress import ProgressBar
import numpy as np
total_loss_record_ganzs = []
mlm_loss_record_ganzs = []
disc_loss_record_ganzs = []
mlm_acc_record_ganzs = []
disc_acc_record_ganzs = []
total_loss_record_electra = []
mlm_loss_record_electra = []
disc_loss_record_electra = []
mlm_acc_record_electra = []
disc_acc_record_electra = []
steps = 80
def plot_figure(file_name, title, t, data1, data2, length, label1, label2, type):
    plt.figure()
    plt.plot(t, data1[:length], label=label1)
    plt.plot(t, data2[:length], label=label2)
    plt.title(title)
    plt.legend()
    plt.xlabel('10 Iterations')
    plt.ylabel(type)
    plt.savefig(file_name)

def parse_line(total_loss, mlm_loss, disc_loss, mlm_acc, disc_acc, line):
    arr = [i for i in line.split(' ') if not i == '']
    if arr[0] == '[1,0]<stdout>:Step:':
        total_loss.append(float(arr[3][:-1]))
        mlm_loss.append(float(arr[5][:-1]))
        disc_loss.append(float(arr[7][:-1]))
        mlm_acc.append(float(arr[9][:-1]))
        disc_acc.append(float(arr[11][:-1]))
        return True
    return False

# Ganzs_progress = ProgressBar(241, description="Ganzs Progress:")        
i = steps
with open('test_output_cleaned_basic') as ganzs:
    # Ganzs_progress()
    # Ganzs_progress.current += 1
    line = ganzs.readline()
    while line and i > 0:
        ret = parse_line(
            total_loss_record_ganzs, 
            mlm_loss_record_ganzs, 
            disc_loss_record_ganzs, 
            mlm_acc_record_ganzs,
            disc_acc_record_ganzs,
            line
        )
        line = ganzs.readline()
        if ret:
            i-=1
# Ganzs_progress.done()

# Electra_progress = ProgressBar(782, description="Ganzs Progress:")        

i = steps
with open('electra_test_log') as electra:
    # Electra_progress()
    # Electra_progress.current += 1
    line = electra.readline()
    while line and i > 0:
        ret = parse_line(
            total_loss_record_electra,
            mlm_loss_record_electra, 
            disc_loss_record_electra, 
            mlm_acc_record_electra, 
            disc_acc_record_electra,
            line
        )
        line = electra.readline()
        if ret:
            i-=1
# Electra_progress.done()
t = np.array(range(len(total_loss_record_ganzs)))
length = len(total_loss_record_ganzs)
plt.figure()
plot_figure(
    "loss_plot_total.png",
    "Total Loss Plot",
    t,
    total_loss_record_ganzs,
    total_loss_record_electra,
    length,
    'GANzs total loss',
    'Electra total loss',
    'Loss'
)

plot_figure(
    "loss_plot_gen.png",
    "Generator Loss Plot",
    t,
    mlm_loss_record_ganzs,
    mlm_loss_record_electra,
    length,
    'GANzs generator loss',
    'Electra generator loss',
    'Loss'
)

plot_figure(
    "loss_plot_disc.png",
    "Discriminator Loss Plot",
    t,
    disc_loss_record_ganzs,
    disc_loss_record_electra,
    length,
    'GANzs discriminator loss',
    'Electra discriminator loss',
    'Loss'
)




plot_figure(
    "acc_plot_gen.png",
    "Generator Acc Plot",
    t,
    disc_acc_record_ganzs,
    disc_acc_record_electra,
    length,
    'GANzs generator acc',
    'Electra generator acc',
    'Acc'
)

plot_figure(
    "acc_plot_disc.png",
    "Discriminator Acc Plot",
    t,
    disc_acc_record_ganzs,
    disc_acc_record_electra,
    length,
    'GANzs discriminator acc',
    'Electra discriminator acc',
    'Acc'
)