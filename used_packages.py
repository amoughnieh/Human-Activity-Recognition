import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
#from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import get_scorer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_auc_score, average_precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import gaussian_filter
import random
import os
import json
import re
from sklearnex import patch_sklearn
import warnings
warnings.filterwarnings('ignore')
