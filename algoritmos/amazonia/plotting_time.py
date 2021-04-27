"""
Convolutional Neural Network to classify Amazon dataset
Plotting the classification time results

Created on TUE Apr 30 2021     10:00:00

@author: micheldearaujo

"""

# Importing the library
from utilities import *

cnn = pd.read_csv(base_dir+'/'+'cnn_scores_03.csv')
knn = pd.read_csv(base_dir+'/'+'knn_scores_all.csv')
rfc = pd.read_csv(base_dir+'/'+'rfc_scores_all.csv')
#cnn.dropna(inplace=True)
#knn.dropna(inplace=True)
#rfc.dropna(inplace=True)

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
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False



    # Classification time


fig, axs = plt.subplots()

# Making plots of the CNN algorithm
axs.set_xlabel('Image Size')
axs.set_ylabel('Single Classifying Time (s)')
axs.set_title('Single Classifying Time for the CNN')
axs.plot(cnn['Target size'],cnn['Single classifying time (s)'], marker='o', markersize=9)
# axs.xaxis.set_major_locator(MultipleLocator(1))
# axs.yaxis.set_major_locator(MultipleLocator(0.5))
# axs.xaxis.set_minor_locator(AutoMinorLocator(1))
# axs.yaxis.set_minor_locator(AutoMinorLocator(1))
axs.grid(which='major', linestyle='--')
axs.grid(which='minor', linestyle=':')


fig2, axs = plt.subplots()
# Making plots of the RFC algorithm
axs.set_xlabel('Image Size')
axs.set_ylabel('Single Classifying Time (s)')
axs.set_title('Single Classifying Time for the RFC')
axs.plot(rfc[rfc['n Trees']==100]['Target size'], rfc[rfc['n Trees']==100]['Single classifying time (s)'], c='orange', label='n Trees = 100', marker='o', markersize=7)
axs.plot(rfc[rfc['n Trees']==500]['Target size'], rfc[rfc['n Trees']==500]['Single classifying time (s)'], label='n Trees = 500', marker='D', markersize=8)
axs.legend()
# axs.xaxis.set_major_locator(MultipleLocator(1))
# axs.yaxis.set_major_locator(MultipleLocator(5))
# axs.xaxis.set_minor_locator(AutoMinorLocator(0.5))
# axs.yaxis.set_minor_locator(AutoMinorLocator(0.5))
axs.grid(which='major', linestyle='--')
axs.grid(which='minor', linestyle=':')


fig3, axs = plt.subplots()
# Making plots of the KNN algorithm
axs.set_xlabel('Image Size')
axs.set_ylabel('Single Classifying Time (s)')
axs.set_title('Single Classifying Time for the KNN')
axs.plot(knn['Target size'], knn['Single classifying time (s)'], marker='o', markersize=9)
axs.xaxis.set_major_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(0.5))
axs.xaxis.set_minor_locator(AutoMinorLocator(1))
axs.yaxis.set_minor_locator(AutoMinorLocator(1))
axs.grid(which='major', linestyle='--')
axs.grid(which='minor', linestyle=':')



plt.tight_layout()
plt.show()
