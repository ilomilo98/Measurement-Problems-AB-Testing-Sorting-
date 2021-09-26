import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################
# Hipotez
############################

# veri setinin incelenmesi:
df_control = pd.read_excel("/ab_testing.xlsx", usecols=[0,1,2,3], sheet_name= "Control Group")
df_test = pd.read_excel("/ab_testing.xlsx", usecols=[0,1,2,3], sheet_name= "Test Group")
df_control.head()
df_test.head()
df_control.describe()
df_test.describe()

df_control.isnull().sum()
df_test.isnull().sum()
############################
# 1. Varsayım Kontrolü
############################

# Normallik Varsayımı
# Varyans Homojenliği

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.
for col in df_control.columns:
    test_stat, pvalue = shapiro(df_control[col])
    if pvalue > 0.05:
        print(col, ':\n H0 cannot be rejected. (Test Stat = {:.2f}, p-value = {:.2f})'.format(test_stat, pvalue))
      else:
        print(col, ':\n H0 rejected. (Test Stat = {:.2f}, p-value = {:.2f})'.format(test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

for col in df_test.columns:
    test_stat, pvalue = shapiro(df_test[col])
    if pvalue > 0.05:
        print(col, '\n H0 cannot be rejected. (Test Stat = {:.2f}, p-value = {:.2f})'.format(test_stat, pvalue))
    else:
        print('\n H0 rejected. (Test Stat = {:.2f}, p-value = {:.2f})'.format(test_stat, pvalue))

############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

for col in df_control.columns:
    test_stat, pvalue = levene(df_control[col], df_test[col])
    if pvalue > 0.05:
        print(col,'\n H0 cannot be rejected.  (Test Stat = {:.2f}, p-value = {:.2f})'.format(test_stat, pvalue))
    else:
        print(col, '\n H0 rejected. (Test Stat = {:.2f}, p-value = {:.2f})'.format(test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


#Sadece 'click' rededilir çıktı

############################
# Hipotezin Uygulanması
############################
# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

# Eğer normallik sağlanmazsa her türlü nonparametrik test yapacağız.
# Eger normallik sağlanır varyans homojenliği sağlanmazsa ne olacak?
# T test fonksiyonuna arguman gireceğiz.

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################
#H0: M1 = M2 (There are no significant difference between the means of the control and test groups.)
#H1: M1 != M2 (There are significant difference between the means of the control and test groups.)

for col in df_control.columns:
    if col == 'Click':
        test_stat, pvalue = ttest_ind(df_control[col], df_test[col], equal_var=False)
        print(col.upper(),"\nMean Values for 2 Groups \n Control: {:.2f} \n Test: {:.2f}".format(df_control[col].mean(), df_test[col].mean()))
        if pvalue > 0.05:
            print('H0 cannot be rejected. (Test Stat = {:.2f}, p-value = {:.2f}) \n'.format(test_stat, pvalue))
        else:
            print('H0 rejected. (Test Stat = {:.2f}, p-value = {:.2f}) \n'.format(test_stat, pvalue))


for col in df_control.columns:
    if col != 'Click':
        print(col.upper(),"\nMean \n Control: {:.2f} \n Test: {:.2f}".format(df_control[col].mean(), df_test[col].mean()))
        test_stat, pvalue = ttest_ind(df_control[col], df_test[col], equal_var=True)
        if pvalue > 0.05:
            print('H0 cannot be rejected. (Test Stat = {:.2f}, p-value = {:.2f}) \n'.format(test_stat, pvalue))
        else:
            print('H0 rejected. (Test Stat = {:.2f}, p-value = {:.2f}) \n'.format(test_stat, pvalue))
