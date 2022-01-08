from matplotlib import pyplot as plt
from progress import ProgressBar
import numpy as np
total_loss_record_ganzs = []
mlm_loss_record_ganzs = []
disc_loss_record_ganzs = []
total_loss_record_electra = []
mlm_loss_record_electra = []
disc_loss_record_electra = []

def plot_figure(file_name, title, t, data1, data2, length, label1, label2):
    plt.figure()
    plt.plot(t, data1[:length], label=label1)
    plt.plot(t, data2[:length], label=label2)
    plt.title(title)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(file_name)

def parse_line(total_loss, mlm_loss, disc_loss, line):
    arr = [i for i in line.split(' ') if not i == '']
    if arr[0] == '[1,0]<stdout>:Step:':
        total_loss.append(float(arr[3][:-1]))
        mlm_loss.append(float(arr[5][:-1]))
        disc_loss.append(float(arr[7][:-1]))

Ganzs_progress = ProgressBar(241, description="Ganzs Progress:")        
with open('test_output_cleaned') as ganzs:
    Ganzs_progress()
    Ganzs_progress.current += 1
    line = ganzs.readline()
    while line:
        parse_line(total_loss_record_ganzs, mlm_loss_record_ganzs, disc_loss_record_ganzs, line)
        line = ganzs.readline()
Ganzs_progress.done()

Electra_progress = ProgressBar(782, description="Ganzs Progress:")        

with open('electra_log') as electra:
    Electra_progress()
    Electra_progress.current += 1
    line = electra.readline()
    while line:
        parse_line(total_loss_record_electra, mlm_loss_record_electra, disc_loss_record_electra, line)
        line = electra.readline()
Electra_progress.done()
t = np.array(range(len(total_loss_record_ganzs)))
length = len(total_loss_record_ganzs)
print(length)
plt.figure()
plot_figure(
    "loss_plot_total.png",
    "Total Loss Plot",
    t,
    total_loss_record_ganzs,
    total_loss_record_electra,
    length,
    'GANzs total loss',
    'Electra total loss'
)

plot_figure(
    "loss_plot_gen.png",
    "Generator Loss Plot",
    t,
    mlm_loss_record_ganzs,
    mlm_loss_record_electra,
    length,
    'GANzs generator loss',
    'Electra generator loss'
)

plot_figure(
    "loss_plot_disc.png",
    "Discriminator Loss Plot",
    t,
    disc_loss_record_ganzs,
    disc_loss_record_electra,
    length,
    'GANzs discriminator loss',
    'Electra discriminator loss'
)