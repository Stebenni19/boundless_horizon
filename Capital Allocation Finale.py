# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 22:16:50 2021

@author: 111949
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.stats import norm

def param_transform(mu, sigma):
    mu_transform = 0.5 * np.log(mu**4 / (mu**2 + sigma**2))
    sigma_transform = np.log(1 + (sigma**2 / mu**2))*0.5
    return mu_transform, sigma_transform

np.random.seed(1234)
#######################
#DEFINING U
#######################
mu_u=2.9
sigma_u=0.4

mu_u_transf, sigma_u_transf= param_transform(mu_u,sigma_u)

#generating u form a uniform
u = np.random.uniform(0,1,10000)

U=pd.Series(lognorm.ppf(u,mu_u_transf,sigma_u_transf))

pd.Series(U).hist(bins=100)

pd.Series(u).hist(bins=100)


#######################
#DEFINING V
#######################
mu_v=2.8
sigma_v=0.5

mu_v_transf, sigma_v_transf= param_transform(mu_v,sigma_v)

#generating u form a uniform
v = np.random.uniform(0,1,10000)

V= pd.Series(lognorm.ppf(v,mu_v_transf,sigma_v_transf))

pd.Series(V).hist(bins=100)

#######################
#DEFINING X
#######################
mu_x=2.7
sigma_x=0.6

mu_x_transf, sigma_x_transf= param_transform(mu_x,sigma_x)

#generating u form a uniform
x = np.random.uniform(0,1,10000)

X= pd.Series(lognorm.ppf(x,mu_x_transf,sigma_x_transf))

X_sorted=pd.Series(X).sort_values().reset_index(drop=True)

pd.Series(X).hist(bins=100)


#######################
#U-V scatterplot
#######################
df=pd.concat([U,V],1)
df.columns=['U','V']
df.sort_values(by='U',inplace=True)
df_rank=df.rank(axis=0, method='first')
plt.scatter(x=df_rank.U, y=df_rank.V, s=0.5)

#######################
#DEFINING Y
#######################

Y=U+V

Y_sorted=pd.Series(Y).sort_values().reset_index(drop=True)
pd.Series(Y).hist(bins=1000)



#######################
#X-Y scatterplot
#######################
df=pd.concat([X,Y],1)
df.columns=['X','Y']
df.sort_values(by='X',inplace=True)
df_rank=df.rank(axis=0, method='first')
plt.scatter(x=df_rank.X, y=df_rank.Y, s=0.5)


#######################
#DEFINING Z
#######################

Z=Y+X

Z_sorted=pd.Series(Z).sort_values().reset_index(drop=True)
pd.Series(Z).hist(bins=1000)



########################################
# 1) CALCULATING THE CAPITAL ALLOCATION TO EACH RISK
########################################

alpha=99

#VaR X
var_99_x= np.percentile(X_sorted, alpha)
#CVaR X
cvar_99_x= X[X>= var_99_x].mean()

X_mean=X.mean()



#VaR U
var_99_u= np.percentile(U, alpha)
#CVaR U
cvar_99_u= U[U>= var_99_u].mean()

U_mean=U.mean()

#VaR V
var_99_v= np.percentile(V, alpha)
#CVaR U
cvar_99_v= V[V>= var_99_v].mean()

V_mean=V.mean()

#VaR z
var_99_z = np.percentile(Z, alpha)
#CVaR z
cvar_99_z = Z[Z >= var_99_z].mean()

#building the risks table
risks_table=pd.concat([Z,X,U,Y,V],1)
risks_table.columns=['Z','X','U','Y','V']

#sorting all the columns according to Z values
risks_table.sort_values('Z',inplace=True)
risks_table.reset_index(drop=True,inplace=True)

#conditional expected shortfall
cvar_99_xz= risks_table[risks_table['Z'] >= var_99_z]['X'].mean()
cvar_99_vz= risks_table[risks_table['Z'] >= var_99_z]['V'].mean()
cvar_99_uz= risks_table[risks_table['Z'] >= var_99_z]['U'].mean()


#capital allocation

C_x= cvar_99_xz - X_mean
C_v= cvar_99_vz - V_mean
C_u= cvar_99_uz - U_mean

#the sum of each capital is the total capital allocated to the portfolio
C_z= C_x+ C_v+ C_u

#note that C_z result is equal to:
C_z_alternative= cvar_99_z -Z.mean()

#computing the capital contribution in percentual terms
perc_X=C_x/C_z
perc_V=C_v/C_z
perc_U=C_u/C_z

#summing the percentage returns 1 
perc_X + perc_U + perc_V


########################################
#PORTFOLIO BENEFIT   as risk measure we use the xTVaR
########################################

#this is useless
#div_benefit_X = 1-(C_x/(cvar_99_x-X.mean())) 
#div_benefit_V = 1-(C_v/(cvar_99_v-V.mean())) 
#div_benefit_U = 1-(C_u/(cvar_99_u-U.mean())) 

#this is the diversification benefit for the portfolio
div_benefit_Z = 1 - (C_z/((cvar_99_x-X.mean()) + 
                           (cvar_99_u-U.mean())+
                          ( cvar_99_v-V.mean())))


########################################
#  2) INTRODUCING DEPENDENCIES VIA SURVIVAL COPULA
########################################

#DEFINING Y with SURVIVAL CLAYTON COPULA 


tetha= 0.5


#in order to create a vector of values that is the result of summation of U and V 
#via a mirrored clayton copula, I have to apply the following transformation.

# following instructions of chapter 2, slides 15-16
u2=(((u)**(-tetha))*(((v)**(-tetha/(tetha+1)))-1)+1)**(-1/tetha)
pd.Series(u2).hist(bins=100)

U2=pd.Series(lognorm.ppf(1-u,mu_u_transf,sigma_u_transf))
U2.hist(bins=100)

V2=pd.Series(lognorm.ppf(1-u2,mu_v_transf,sigma_v_transf))
V2.hist(bins=100)


#rank scatter
df_survival=pd.concat([U2,V2],1)
df_survival.columns=['U2','V2']
df_survival.sort_values(by='U2', inplace=True)
df_survival_rank=df_survival.rank(axis=0, method='first')
plt.scatter(x=df_survival_rank.U2, y=df_survival_rank.V2, s=0.5)


Y_survival= V2 + U2
Y_survival.hist(bins=100)

Y_surv_rank=Y_survival.rank(method='first',ascending=False)
unif_Y=Y_surv_rank/len(Y_surv_rank)

unif_Y.hist()

# now we aggregate X and Y_survival via a mirrored clayton copula of parameter tetha= 1
# To create the final portfolio I will sum the new vector of values X2 and X.
tetha= 1

x2=(((unif_Y)**(-tetha))*(((x)**(-tetha/(tetha+1)))-1)+1)**(-1/tetha)

X2=pd.Series(lognorm.ppf(1-x2,mu_x_transf,sigma_x_transf))
X2.hist(bins=100)
#plt.scatter(x=x2, y=unif_Y.rank(method='first'), s=0.5)


#this is my final portfolio
Z_survival=X2+Y_survival



df_survival_xy=pd.concat([X2,Y_survival],1)
df_survival_xy.columns=['X2','Y_survival']
df_survival_xy.sort_values(by='X2', inplace=True)
df_survival_rank_xy=df_survival_xy.rank(axis=0, method='first')
plt.scatter(x=df_survival_rank_xy.Y_survival, y=df_survival_rank_xy.X2, s=0.5)

########################################
# 2a) CALCULATING THE CAPITAL ALLOCATION TO EACH RISK
########################################

alpha=99

#VaR X
var_99_x2= np.percentile(X2, alpha)
#CVaR X
cvar_99_x2= X2[X2>= var_99_x2].mean()

X2_mean=X2.mean()


#VaR U
var_99_u2= np.percentile(U2, alpha)
#CVaR U
cvar_99_u2= U2[U2>= var_99_u2].mean()

U2_mean=U2.mean()


#VaR V
var_99_v2= np.percentile(V2, alpha)
#CVaR U
cvar_99_v2= V2[V2>= var_99_v2].mean()

V2_mean=V2.mean()

#VaR z
var_99_z2 = np.percentile(Z_survival, alpha)
#CVaR z
cvar_99_z2 = Z_survival[Z_survival >= var_99_z2].mean()

#building the risks table
risks_table_surv=pd.concat([Z_survival,X2,U2,V2],1)
risks_table_surv.columns=['Z2','X2','U2','V2']

#sorting all the columns according to Z2 values
risks_table_surv.sort_values('Z2',inplace=True)
risks_table_surv.reset_index(drop=True,inplace=True)

#conditional expected shortfall
cvar_99_xz_surv= risks_table_surv[risks_table_surv['Z2'] >= var_99_z2]['X2'].mean()
cvar_99_vz_surv= risks_table_surv[risks_table_surv['Z2'] >= var_99_z2]['V2'].mean()
cvar_99_uz_surv= risks_table_surv[risks_table_surv['Z2'] >= var_99_z2]['U2'].mean()


#capital allocation

C_x2= cvar_99_xz_surv - X2_mean
C_v2= cvar_99_vz_surv - V2_mean
C_u2= cvar_99_uz_surv - U2_mean

#the sum of each capital is the total capital allocated to the portfolio
C_z2= C_x2+ C_v2+ C_u2

#computing the capital contribution in percentual terms
perc_X2=C_x2/C_z2
perc_V2=C_v2/C_z2
perc_U2=C_u2/C_z2

#note that C_z result is equal to:
C_z_alternative= cvar_99_z2 -Z_survival.mean()

#summing the percentage returns 1 
perc_X2 + perc_U2 + perc_V2


########################################
#PORTFOLIO BENEFIT   as risk measure we use the xTVaR
########################################

#this is the diversification benefit for the portfolio
div_benefit_Z_surv = +1 - (C_z2/((cvar_99_x2-X2.mean()) + 
                           (cvar_99_v2-V2.mean())+
                          (cvar_99_u2-U2.mean())))




###############################################################################
#3) GAUSSIAN COPULA
###############################################################################

#drawing samples from normal standard
U_norm=pd.Series(np.random.normal(0, 1, 10000))
V_norm=pd.Series(np.random.normal(0, 1, 10000))
X_norm=pd.Series(np.random.normal(0, 1, 10000))

#concatenating V and U obtained at problem number 2 so that we can get the corr matrix
#to decompose with cholesky.
V_U_df=pd.concat([V2,U2],1)
V_U_df.columns=['V','U']

#computing rank correlation (Spearman's Correlation)
corr=V_U_df.corr(method='spearman')
chol_decon=np.linalg.cholesky(corr)

#concatenating new draws form normal standard
u_v_df=pd.concat([U_norm,V_norm],1)
u_v_df.columns=['U','V']

new_variables=np.matmul(u_v_df,chol_decon)

new_variables.iloc[:,0].hist(bins=100)

#transforming v,u from a normal to a uniform
V_norm_unif=norm.cdf(new_variables['V'])
U_norm_unif=norm.cdf(new_variables['U'])

#concatenating V uniform and U uniform
unif_concat=pd.concat([pd.Series(V_norm_unif), pd.Series(U_norm_unif)],1)
unif_concat.columns=['V','U']

#Ranking values and plot the scatter
unif_concat_rank=unif_concat.rank(axis=0, method='first')
unif_concat_rank.sort_values(by='V', inplace=True)
plt.scatter(x=unif_concat_rank.V, y=unif_concat_rank.U, s=0.5)


#using U and V to create the lognormals
U_gauss=pd.Series(lognorm.ppf(U_norm_unif,mu_u_transf,sigma_u_transf))
V_gauss=pd.Series(lognorm.ppf(V_norm_unif,mu_v_transf,sigma_v_transf))

#obtaining the Y portfolio
Y_gauss=U_gauss+V_gauss
#log gauss
Y_gauss.hist(bins=100)
#Y_gauss back to normal dist in order to compute gauss copula again
(np.log(Y_gauss)).hist(bins=100)
Y_gauss_norm=np.log(Y_gauss)


#computing correlation btw X2 and Y_survival (from point 2)
X_Y_df=pd.concat([X2,Y_survival],1)
X_Y_df.columns=['X2','Y_surv']

#computing rank correlation (Spearman's Correlation)
corr=X_Y_df.corr(method='spearman')
chol_decon=np.linalg.cholesky(corr)

#concatenating new draws form normal standard
Xnor_Ynor_df=pd.concat([X_norm,Y_gauss_norm],1)
Xnor_Ynor_df.columns=['X_nor','Y_gauss_norm']

new_variables=np.matmul(Xnor_Ynor_df,chol_decon)

new_variables.iloc[:,0].hist(bins=100)


#transforming x from a normal to a uniform
X_norm_unif=norm.cdf(new_variables['X_nor'])



X_gauss=pd.Series(lognorm.ppf(X_norm_unif,mu_x_transf,sigma_x_transf))

x_y_concat=pd.concat([X_gauss,Y_gauss],1)
x_y_concat.columns=['X','Y']
x_y_rank=x_y_concat.rank(axis=0, method='first')
x_y_rank.sort_values(by='X',inplace=True)
plt.scatter(x=x_y_rank.X, y=x_y_rank.Y, s=0.5)


#the final portfolio is:
Z_gauss=X_gauss+Y_gauss


########################################
# 3b) CALCULATING THE CAPITAL ALLOCATION TO EACH RISK
########################################

alpha=99

#VaR X
var_99_x_gauss= np.percentile(X_gauss, alpha)
#CVaR X
cvar_99_x_gauss= X_gauss[X_gauss>= var_99_x_gauss].mean()

X_gauss_mean=X_gauss.mean()


#VaR U
var_99_u_gauss= np.percentile(U_gauss, alpha)
#CVaR U
cvar_99_u_gauss= U_gauss[U_gauss>= var_99_u_gauss].mean()

U_gauss_mean=U_gauss.mean()


#VaR V
var_99_v_gauss= np.percentile(V_gauss, alpha)
#CVaR U
cvar_99_v_gauss= V_gauss[V_gauss>= var_99_v_gauss].mean()

V_gauss_mean=V_gauss.mean()

#VaR z
var_99_z_gauss = np.percentile(Z_gauss, alpha)
#CVaR z
cvar_99_z_gauss = Z_gauss[Z_gauss >= var_99_z_gauss].mean()

Z_gauss_mean=Z_gauss.mean()

#building the risks table
risks_table_gauss=pd.concat([Z_gauss,X_gauss,U_gauss,V_gauss],1)
risks_table_gauss.columns=['Z','X','U','V']

#sorting all the columns according to Z2 values
risks_table_gauss.sort_values('Z',inplace=True)
risks_table_gauss.reset_index(drop=True,inplace=True)

#conditional expected shortfall
cvar_99_xz_gauss= risks_table_gauss[risks_table_gauss['Z'] >= var_99_z_gauss]['X'].mean()
cvar_99_vz_gauss= risks_table_gauss[risks_table_gauss['Z'] >= var_99_z_gauss]['V'].mean()
cvar_99_uz_gauss= risks_table_gauss[risks_table_gauss['Z'] >= var_99_z_gauss]['U'].mean()


#capital allocation

C_x_gauss= cvar_99_xz_gauss - X_gauss_mean
C_v_gauss= cvar_99_vz_gauss - V_gauss_mean
C_u_gauss= cvar_99_uz_gauss - U_gauss_mean

#the sum of each capital is the total capital allocated to the portfolio
C_z_gauss= C_x_gauss+ C_v_gauss+ C_u_gauss

#computing the capital contribution in percentual terms
perc_X_gauss=C_x_gauss/C_z_gauss
perc_V_gauss=C_v_gauss/C_z_gauss
perc_U_gauss=C_u_gauss/C_z_gauss

#note that C_z result is equal to:
C_z_alternative= cvar_99_z_gauss -Z_gauss.mean()

#summing the percentage returns 1 
perc_X2 + perc_U2 + perc_V2


########################################
#PORTFOLIO BENEFIT   as risk measure we use the xTVaR
########################################

#this is the diversification benefit for the portfolio
div_benefit_Z_gauss = +1 - (C_z_gauss/((cvar_99_x_gauss-X_gauss.mean()) + 
                           (cvar_99_v_gauss-V_gauss.mean())+
                          (cvar_99_u_gauss-U_gauss.mean())))