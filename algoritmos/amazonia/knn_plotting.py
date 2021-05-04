"""
K-Nearest Neighbors to classify Amazon dataset
Plotting the knn results
Created on TUE Apr 30 2021     10:00:00

@author: micheldearaujo

"""
from utilities import *
# the best size is 64x64

# Defining the hyperams
knn = pd.read_csv(base_dir+'/'+'knn_scores_all.csv')

markers =['o','*','v','s','X','D','+','>','p']
sizes=['8x8','16x16','32x32','64x64']
linestyles=['dashed','solid','dashdot','dotted']


SMALL_SIZE = 22
MEDIUM_SIZE = 22
BIGGER_SIZE = 22
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig0, axs = plt.subplots(1)
axs.set_xlabel('Recall')
axs.set_ylabel('Precision')
#axs.set_title('Precision and Recall As Function of Image Size and Threshold for KNN')
#axs.plot(knn['Avg Recall'],
#         knn['Avg Precision'])

for k in range(len(sizes)):
    axs.scatter(knn[knn['Target size'] == sizes[k]]['Avg Recall'],
                knn[knn['Target size'] == sizes[k]]['Avg Precision'],
                marker=markers[k],
                label=sizes[k], s=110)

axs.grid(which='major', linestyle='--')
plt.xlim(0.535,0.555)
axs.grid(which='minor', linestyle=':')
axs.legend(title='Image size')
plt.show()