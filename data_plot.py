import numpy as np
import matplotlib
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


font = {'weight': 'normal',
            'color': 'black',
            'size': 13
            }

def plot_best_fit():
    PSO_best_data = np.loadtxt('img_txt\\pareto_fitness.txt')
    plt.figure(figsize=(3, 2.5), dpi=300)
    plt.scatter(PSO_best_data[:,0]*100,PSO_best_data[:,1],s=10,c='black')
    plt.xlabel('Coupling efficiency(%)',fontdict=font)
    plt.ylabel('Tolerance ($\mu$m)',fontdict=font)
    # plt.ylim([0.7,0.8])
    plt.show()

def plot_PSO():
    PSO_position = np.loadtxt('img_txt\\pareto_in.txt')
    PSO_efficience = np.loadtxt('img_txt\\pareto_fitness.txt')
    plt.figure(figsize=(3, 2.5), dpi=300)
    for i,marker in zip(range(len(PSO_position[0])),['.',',', 'o','v','^']):
        plt.scatter(PSO_efficience[:,0]*100,PSO_position[:,i],s=10,marker=marker)
    plt.xlabel('Coupling efficiency(%)',fontdict=font)
    plt.ylabel('position ($\mu$m)',fontdict=font)
    plt.legend(['PQ','NO','BD','FH','ky'],fontsize=7)
    # plt.ylim([0.7,0.8])
    plt.show()

def plot_sim_data():
    Si = np.loadtxt('data\\Si_waveguide.txt')
    SiN = np.loadtxt('data\\SiN_waveguide.txt')

    plt.figure(figsize=(3, 2.5), dpi=300)
    plt.plot(Si[:,0], Si[:,1]*100,'-k')
    plt.plot(Si[:, 2], Si[:, 3]*100,'-.k')
    plt.legend(['vertical','horizontal'])
    plt.xlabel('position ($\mu$m)',fontdict=font)
    plt.ylabel('coupling efficency(%)',fontdict=font)
    plt.title('Si waveguide',fontdict=font)
    plt.xlim([-2,2])
    plt.ylim([0,85])
    plt.show()

    plt.figure(figsize=(3, 2.5), dpi=300)
    plt.plot(SiN[:, 0], SiN[:, 1]*100, '-k')
    plt.plot(SiN[:, 2], SiN[:, 3]*100, '-.k')
    plt.legend(['vertical', 'horizontal'])
    plt.xlabel('position ($\mu$m)', fontdict=font)
    plt.ylabel('coupling efficency(%)', fontdict=font)
    plt.title('SiN waveguide', fontdict=font)
    plt.xlim([-2, 2])
    plt.ylim([0, 85])
    plt.show()


def plot_three_waveguide():
    PSO = np.loadtxt('data\\three_waveguide.txt')

    plt.figure(figsize=(3, 2.5),dpi=300)
    plt.plot(PSO[:,0],PSO[:,1]*100, '-k')
    plt.plot(PSO[:, 2], PSO[:, 3]*100, '-.k')
    plt.xlabel('position ($\mu$m)',fontdict=font)
    plt.ylabel('coupling efficency(%)',fontdict=font)
    # plt.title('PSO')
    plt.legend(['horizontal','vertical'])
    plt.xlim([-1.5, 1.5])
    plt.ylim([20, 80])
    plt.show()

    PSO = np.loadtxt('data\\three_waveguide2.txt')

    plt.figure(figsize=(3, 2.5), dpi=300)
    plt.plot(PSO[:, 0], PSO[:, 1] * 100, '-k')
    plt.plot(PSO[:, 2], PSO[:, 3] * 100, '-.k')
    plt.xlabel('position ($\mu$m)', fontdict=font)
    plt.ylabel('coupling efficency(%)', fontdict=font)
    # plt.title('PSO')
    plt.legend(['horizontal', 'vertical'])
    plt.xlim([-1.5, 1.5])
    plt.ylim([20, 80])
    plt.show()



def lab_data(data_sim,data_lab,title):
    x = data_sim[:, 0]
    y = data_sim[:, 1]/np.max(data_sim[:, 1])
    data_lab[:,1] = 10 ** (data_lab[:,1] / 10)
    data_lab[:,1] = data_lab[:,1] / np.max(data_lab[:,1])
    data_lab[:,1] = 10 * np.log10(data_lab[:,1])

    # z1 = np.polyfit(data_lab[:,0], data_lab[:,1], 3)
    # p1 = np.poly1d(z1)
    # yvals = p1(data_lab[:,0])  # 拟合后的y值

    xnew = data_lab[:,0]
    ynew = data_lab[:,1]
    x_max = xnew[np.where(
        ynew[:int(len(ynew) / 2)] == min(ynew[:int(len(ynew) / 2)], key=lambda x: abs(x - max(ynew)+1)))]
    x_min = xnew[int(len(ynew) / 2) + np.where(
        ynew[int(len(ynew) / 2):] == min(ynew[int(len(ynew) / 2):], key=lambda x: abs(x - max(ynew) +1)))[0]]
    tolerance_1dB = (x_max - x_min) / 2
    print('1dB对准容差为：'+str(tolerance_1dB[0]))

    plt.figure(figsize=(3, 2.5), dpi=300)
    plt.plot(x, 10*np.log10(y),'k',linewidth=1.5)
    plt.plot(data_lab[:, 0], data_lab[:,1],'ro',markersize=3)
    plt.xlabel('position ($\mu$m)',fontdict=font)
    plt.ylabel('relative optical intensity(dB)',fontdict=font)
    plt.legend(['simulation','experiment'])
    # plt.title(title)
    plt.xlim([-2,2])
    plt.ylim([-15,1])
    plt.show()


def plot_lab_data():
    Si = np.loadtxt('data\\Si_waveguide.txt')
    SiN = np.loadtxt('data\\SiN_waveguide.txt')

    Si_v_lab = np.loadtxt('data\\Si_vertical_shift_lab.txt')
    Si_h_lab = np.loadtxt('data\\Si_horizontal_shift_lab.txt')
    SiN_v_lab = np.loadtxt('data\\SiN_vertical_shift_lab.txt')
    SiN_h_lab = np.loadtxt('data\\SiN_horizontal_shift_lab.txt')

    SiN_v_lab[:,1]

    lab_data(SiN[:,:2],SiN_v_lab,'SiN vertical alignment tolerance')
    lab_data(SiN[:,2:],SiN_h_lab,'SiN horizontal alignment tolerance')
    lab_data(Si[:,:2],Si_v_lab,'Si vertical alignment tolerance')
    lab_data(Si[:,2:],Si_h_lab,'Si horizontal alignment tolerance')

# plot_best_fit()
plot_PSO()
# plot_sim_data()
# plot_three_waveguide()
# plot_lab_data()
