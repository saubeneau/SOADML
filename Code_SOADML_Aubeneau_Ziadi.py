# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:57:29 2018

@author: Simon
"""
import numpy as np
import random
import time
from matplotlib import pyplot

##############################################
####    Implémentation des algorithmes    ####
##############################################

### Stochastic Gradient Descent (SGD)
def SGD (X, y, w_zero, n, d, lambda_, arret, aff, d_aff):
    w = w_zero.copy()
    
    hinge_loss = []
    class_err = []
    min_problem = []
    label_time = []
    label_iter = []
    
    # Initialisation du learning parameter
    eta = 1
    
    # on parcours nb_epoch fois tout le dataset.
    # On commence l'itération à 1 pour éviter la division par zéro durant le calcul du paramètre de régularisation.
    t = 0
    init_time = time.time()
    aff_time = 0
    
    while not arret(t, time.time()-init_time-aff_time):
        
        eta = 1 / (t+1)
        
        i=random.randint(0,n-1)
        # condition de mauvaise classification yi⟨xi,w⟩<1
        if (y[i]*np.dot(X[:,i], w)) < 1:
            # Règle de mise à jour pour les poids w = w + η (yixi-2λw) incluant le taux d'apprentissage eta et le régularisateur λ
            w = w + eta * ( (X[:,i] * y[i]) + (-2  *lambda_* w) )
        else:
            # S'il est classé correctement, il suffit de mettre à jour w par le terme dérivé du régularisateur w = w + η (-2λw).
            w = w + eta * (-2  *lambda_* w)
                
        # calcul de la hinge loss et de l'erreur de classification
        if( t%d_aff == 0 and aff):
            time_temp = time.time()
            
            b = compute_b(X, y, w, n, d)
            y_pred = prediction(w, b, n, d, X)
            class_err.append(erreur_classification(y, y_pred, n)[0])
                
            #hinge_loss.append(hinge_loss_sum(w, X, y, n))
            
            min_problem.append(minimization_problem(w, X, y, n, lambda_))
            
            label_time.append(time.time()-init_time-aff_time)
            label_iter.append(t/n)
            
            aff_time += time.time() - time_temp
            
        t += 1

    return { 'w':w, 'hinge_loss':hinge_loss, 'class_err':class_err, 'min_problem':min_problem, 'label_time':label_time, 'label_iter':label_iter }


###  SDCA (problème primal-dual)
def SDCA(X, y, alpha_zero, w_zero, n, d, lambda_, option_output,  arret, aff, d_aff):
    alpha = alpha_zero.copy() 
    
    alpha = np.zeros((n,1000000))
    w = np.zeros((d,1000000))
    w[:,0] = w_zero.copy()
    
    hinge_loss = []
    class_err = []
    min_problem = []
    label_time = []
    label_iter = []
    
    # on calcule alpha et w à chaque itération
    # le nombre de fois qu'on parcours le jeu de données
    t = 1
    init_time = time.time()
    aff_time = 0
    
    while not arret(t, time.time()-init_time-aff_time):
 
        i = np.random.randint(n)
        
        # calcul de delta_alpha
        xw = X[:,i].transpose().dot(w[:,t-1])
        num = 1 - xw*y[i]
        denom = np.linalg.norm(X[:,i])**2/(lambda_*n)
        v = num / denom + alpha[i,t-1]*y[i]
        delta_alpha = y[i]*max(0,min(1,v))-alpha[i,t-1]
        # mise à jour des paramètres 
        alpha[:,t] = alpha[:,t-1]
        alpha[i,t] = alpha[i,t] + delta_alpha
        w[:,t] = w[:,t-1] + (delta_alpha*X[:,i])/(lambda_*n)
        
        # calcul de la hinge loss et de l'erreur de classification
        if( t%d_aff == 0 and aff):
            time_temp = time.time()
            
            b = compute_b(X, y, w[:,t], n, d)
            y_pred = prediction(w[:,t], b, n, d, X)
            class_err.append(erreur_classification(y, y_pred, n)[0])
                
            hinge_loss.append(hinge_loss_sum(w[:,t], X, y, n))
            
            min_problem.append(minimization_problem(w[:,t], X, y, n, lambda_))
            
            label_time.append(time.time()-init_time-aff_time)
            label_iter.append(t/n)
            
            aff_time += time.time() - time_temp
        
        t += 1

    T=t
    
    # on calcule le alpha et le w final         
    # Option Average
    if option_output == 'Average' :
        sum_alpha = 0 
        for t in range(T//2+1,T):
            sum_alpha = sum_alpha + alpha[:,t-1]
        alpha = sum_alpha/ ( T - T/2)
        sum_w = 0 
        for t in range(T//2+1,T):
            sum_w = sum_w + w[:,t-1]
        w = sum_w  / ( T - T/2)
    # Option Random
    if option_output == 'Random' :
        rand = np.random.randint(T/2, T) 
        w = w[:,rand]
        alpha = alpha[:,rand] 
        
    return { 'w':w, 'alpha':alpha, 'hinge_loss':hinge_loss, 'class_err':class_err, 'min_problem':min_problem, 'label_time':label_time, 'label_iter':label_iter }   



###  SDCA Perm (problème primal-dual)
def SDCA_perm(X, y, alpha_zero, w_zero, n, d, lambda_, option_output,  arret, aff, d_aff) :
    alpha = alpha_zero.copy() 
    
    alpha = np.zeros((n,1000000))
    w = np.zeros((d,1000000))
    w[:,0] = w_zero.copy()
    
    hinge_loss = []
    class_err = []
    min_problem = []
    label_time = []
    label_iter = []
    
    # on calcule alpha et w à chaque itération
    # le nombre de fois qu'on parcours le jeu de données
    t = 1
    init_time = time.time()
    aff_time = 0
    
    while not arret(t, time.time()-init_time-aff_time): # le nombre de fois qu'on parcours le jeu de données
        perm = np.array(range(n)) # permutation des indices des données
        np.random.shuffle(perm) 
        for j in range(n): 
            i = perm[j] # on parcours le vecteur permuté
            
            # calcul de delta_alpha
            xw = X[:,i].transpose().dot(w[:,t-1])
            num = 1 - xw*y[i]
            denom = np.linalg.norm(X[:,i])**2/(lambda_*n)
            v = num / denom + alpha[i,t-1]*y[i]
            delta_alpha = y[i]*max(0,min(1,v))-alpha[i,t-1]
            # mise à jour des paramètres 
            alpha[:,t] = alpha[:,t-1]
            alpha[i,t] = alpha[i,t] + delta_alpha
            w[:,t] = w[:,t-1] + (delta_alpha*X[:,i])/(lambda_*n)
                
            # calcul de la hinge loss et de l'erreur de classification
            if( t%d_aff == 0 and aff):
                time_temp = time.time()
                
                b = compute_b(X, y, w[:,t], n, d)
                y_pred = prediction(w[:,t], b, n, d, X)
                class_err.append(erreur_classification(y, y_pred, n)[0])
                    
                hinge_loss.append(hinge_loss_sum(w[:,t], X, y, n))
            
                min_problem.append(minimization_problem(w[:,t], X, y, n, lambda_))
            
                label_time.append(time.time()-init_time-aff_time)
                label_iter.append(t/n)
                
                aff_time += time.time() - time_temp
            
            t += 1
            if( arret(t, time.time()-init_time-aff_time)):
                break
             
    
    T=t
    
    # Option Average
    if option_output == 'Average' :
        sum_alpha = 0 
        for t in range(T//2+1,T):
            sum_alpha = sum_alpha + alpha[:,t-1]
        alpha = sum_alpha/ ( T - T/2)
        sum_w = 0 
        for t in range(T//2+1,T):
            sum_w = sum_w + w[:,t-1]
        w = sum_w  / ( T - T/2)
    # Option Random
    if option_output == 'Random' :
        rand = np.random.randint(T/2, T) 
        w = w[:,rand]
        alpha = alpha[:,rand] 
        
    return { 'w':w, 'alpha':alpha, 'hinge_loss':hinge_loss, 'class_err':class_err, 'min_problem':min_problem, 'label_time':label_time, 'label_iter':label_iter}

### Pegasos
def pegasos(X, y, w_zero, n, d, lambda_, arret, aff, d_aff):
    w = w_zero.copy()
    hinge_loss = []
    class_err = []
    min_problem = []
    label_time = []
    label_iter = []
    init_time = time.time()
    aff_time = 0
    t = 0
    while not arret(t, time.time()-init_time-aff_time):
        i=random.randint(0,n-1)
        n_t = 1/(lambda_*(1+t))
        if( y[i]*(w.transpose().dot(X[:,i])) < 1):
            w = (1 - n_t * lambda_)*w + n_t*y[i]*X[:,i]
        else:
            w = (1 - n_t * lambda_)*w
            
         # calcul de la hinge loss et de l'erreur de classification
        if( t%d_aff == 0 and aff):
            time_temp = time.time()
            
            b = compute_b(X, y, w, n, d)
            y_pred = prediction(w, b, n, d, X)
            class_err.append(erreur_classification(y, y_pred, n)[0])
                
            hinge_loss.append(hinge_loss_sum(w, X, y, n))
            
            min_problem.append(minimization_problem(w, X, y, n, lambda_))
            
            label_time.append(time.time()-init_time-aff_time)
            label_iter.append(t/n)
            
            aff_time += time.time() - time_temp
        
        t += 1
            
    return {'w': w, 'hinge_loss':hinge_loss, 'class_err':class_err, 'min_problem':min_problem, 'label_time':label_time, 'label_iter':label_iter }

### Pegasos Mini-Batch
def pegasos_batch(X, y, w_zero, n, d, lambda_, arret, aff, d_aff, m=50):
    w = w_zero.copy()
    hinge_loss = []
    class_err = []
    min_problem = []
    label_time = []
    label_iter = []
    init_time = time.time()
    aff_time = 0
    t = 0
    while not arret(t, time.time()-init_time-aff_time):
        k = random.randint(1,m)
        A_t = random.sample(range(n), k)
        A_t_plus = []
        for i in A_t:
            if( y[i]*(w.transpose().dot(X[:,i])) < 1 ):
                A_t_plus.append(i)
        n_t = 1/(lambda_*(1+t))
        temp = np.zeros(d)
        for i in A_t_plus:
            temp += y[i]*X[:,i]
        temp *= n_t / k
        w = (1 - n_t * lambda_)*w + temp
        
        if( t%d_aff == 0 and aff):
            time_temp = time.time()
            
            b = compute_b(X, y, w, n, d)
            y_pred = prediction(w, b, n, d, X)
            class_err.append(erreur_classification(y, y_pred, n)[0])
                
            hinge_loss.append(hinge_loss_sum(w, X, y, n))
            
            min_problem.append(minimization_problem(w, X, y, n, lambda_))
            
            label_time.append(time.time()-init_time-aff_time)
            label_iter.append(t/n)
            
            aff_time += time.time() - time_temp
        
        t += 1
        
    return {'w': w, 'hinge_loss':hinge_loss, 'class_err':class_err, 'min_problem':min_problem, 'label_time':label_time, 'label_iter':label_iter }

### Pegasos Kernel
def kernelized_pegasos(X, y, alpha_zero, n, d, lambda_, kernel, arret, aff, d_aff):
    alpha = alpha_zero.copy()
    hinge_loss = []
    class_err = []
    label_time = []
    label_iter = []
    init_time = time.time()
    aff_time = 0
    t = 0
    while not arret(t, time.time()-init_time-aff_time):
        i=random.randint(0,n-1)
        s = 0
        for j in range(n):
            s += alpha[j]*y[i]*kernel(X[:,i],X[:,j])
        s *= y[i]/lambda_
        if( s < 1):
            alpha[i] = alpha[i]+1
            
        if( t%d_aff == 0 and aff):
            if( aff):
                time_temp = time.time()
                
                b = compute_b_kernelized(X, y, alpha, n, d, kernel)
                y_pred = prediction_kernelized(kernel, alpha, X, y, b, n, d, X)
                class_err.append(erreur_classification(y, y_pred, n)[0])
                
                #hinge_loss.append(hinge_loss_sum_kernelized(X, y, n, alpha, kernel))
            
                label_time.append(time.time()-init_time-aff_time)
                label_iter.append(t/n)
                
                aff_time += time.time() - time_temp
        t += 1
    return {'alpha':alpha, 'hinge_loss':hinge_loss, 'class_err':class_err, 'label_time':label_time, 'label_iter':label_iter }

##############################################
####          Calcul des b et w           ####
##############################################
#### calcul de w à partir de alpha
def compute_w(alpha, X, y, n, d):
    w = np.zeros(d)
    for i in range(n):
        w += (y[i]*alpha[i]) * X[:,i]
    return w

#### calcul de b à partir w
def compute_b(X, y, w, n, d):
    v_min = 0
    i_min = -1
    v_max = 0
    i_max = -1
    for i in range(n):
        v = w.transpose().dot(X[:,i])
        #if alpha[i]>0 and y[i]>0 and (v < v_min or i_min==-1):
        if y[i]>0 and (v < v_min or i_min==-1):
            i_min = i
            v_min = v
        #if alpha[i]>0 and y[i]<0 and (v > v_max or i_max==-1):
        if y[i]<0 and (v > v_max or i_max==-1):
            i_max = i
            v_max = v
    b = -0.5*(v_min+v_max)
    return b

#### calcul de b dans le cas où on a un kernel
def compute_b_kernelized(X, y, alpha, n, d, kernel):
    v_min = 0
    i_min = -1
    v_max = 0
    i_max = -1
    for i in range(n):
        v = 0
        for j in range(n):
            v += y[j]*alpha[j]*kernel(X[:,j], X[:,i])
        if alpha[i]>0 and y[i]>0 and (v < v_min or i_min==-1):
            i_min = i
            v_min = v
        if alpha[i]>0 and y[i]<0 and (v > v_max or i_max==-1):
            i_max = i
            v_max = v
    b = -0.5*(v_min+v_max)
    return b

#################################################
####    Calcul des fonctions de décision     ####
#################################################
def decision_function(w, b, n, d, x):
    if( w.dot(x)+b > 0 ):
        return 1
    else:
        return -1
    
def prediction(w, b, n, d, X_test):
    y_pred = np.zeros(n)
    for i in range(n):
        y_pred[i] = decision_function(w, b, n, d, X_test[:,i])
    return y_pred

def decision_function_kernelized( kernel, alpha, X, y, b, n, d, x):
    v = 0
    for j in range(n):
        v += y[j]*alpha[j]*kernel(X[:,j], x)
    if( v+b > 0 ):
        return 1
    else:
        return -1
    
def prediction_kernelized(kernel, alpha, X, y, b, n, d, X_test):
    y_pred = np.zeros(n)
    for i in range(n):
        y_pred[i] = decision_function_kernelized( kernel, alpha, X, y, b, n, d, X_test[:,i])
    return y_pred

##############################################
####          Calcul des erreurs          ####
##############################################
def erreur_classification(y1, y2, n):
    egaux = (y1==y2).sum()
    return( (n-egaux)/n, egaux/n)

def hinge_loss(w, xi, yi):
    return max( 1 - yi * (w.dot(xi)), 0)

def hinge_loss_sum(w, X, y, n):
    s = 0
    for i in range(n):
        s += hinge_loss(w, X[:,i], y[i])
    return s

# Fonction P du problème Primal
def minimization_problem(w, X, y, n, lambda_):
    return hinge_loss_sum(w, X, y, n)/n+(lambda_/2)*(np.linalg.norm(w)**2)

def w_scal_phi_x(x, X, y, alpha, kernel, n):
    s = 0
    for i in range(n):
        s += y[i]*alpha[i]*kernel(X[:,i], x)
    return s
    
def hinge_loss_kernelized(xi, yi, y, X, alpha, kernel, n):
    return max( 1 - yi * (w_scal_phi_x(xi, X, y, alpha, kernel, n)), 0)
    
def hinge_loss_sum_kernelized(X, y, n, alpha, kernel):
    s = 0
    for i in range(n):
        s += hinge_loss_kernelized(X[:,i], y[i], y, X, alpha, kernel, n)
    return s

##############################################
####       Définition des noyaux          ####
##############################################
def gaussian_kernel(x1, x2):
    return np.exp(-np.linalg.norm(x1-x2)**2)

def linear_kernel(x1, x2):
    return x1.dot(x2)

##############################################
####             Visualisation            ####
##############################################
def visualise( X, y, w, b, n ):
    import matplotlib.pyplot as plt 
    x_pos = []
    y_pos = []
    x_neg = []
    y_neg = []
    for i in range(n):
        if( y[i]==1):
            x_pos.append(X[0,i])
            y_pos.append(X[1,i])
        else:
            x_neg.append(X[0,i])
            y_neg.append(X[1,i])
    
    plt.plot(x_pos,y_pos,'o',color='blue',markersize=1)
    plt.plot(x_neg,y_neg,'o',color='red', markersize=1)
    xx = np.linspace(-1.5, 1.5, 100)
    w1 = w[0]
    w2 = w[1]
    plt.plot(xx, - b/w2 - w1*xx/w2,color='black');
    plt.show()

def visualise2( X, y, y_pred ):
    import matplotlib.pyplot as plt 
    x_pos = []
    y_pos = []
    x_neg = []
    y_neg = []
    x_pos_err = []
    y_pos_err = []
    x_neg_err = []
    y_neg_err = []
    for i in range(n):
        if( y[i]==1):
            if( y_pred[i]==1):
                x_pos.append(X[0,i])
                y_pos.append(X[1,i])
            else:
                x_pos_err.append(X[0,i])
                y_pos_err.append(X[1,i])
        else:
            if( y_pred[i]==-1):
                x_neg.append(X[0,i])
                y_neg.append(X[1,i])
            else:
                x_neg_err.append(X[0,i])
                y_neg_err.append(X[1,i])
    
    plt.plot(x_pos,y_pos,'o',color='blue',markersize=1)
    plt.plot(x_neg,y_neg,'o',color='red', markersize=1)
    plt.plot(x_pos_err,y_pos_err,'o',color='cyan', markersize=1)
    plt.plot(x_neg_err,y_neg_err,'o',color='orange', markersize=1)
    plt.show()


##############################################
####        Génération des donnéés        ####
##############################################
def initialisation(n, d):
    alpha_zero = np.zeros( n )
    w_zero = np.zeros(d)
    return (alpha_zero, w_zero)

# données linéairement séparables
def generer_lineaire():
    d = 2
    n = 2000
    
    listX = []
    for i in range(n):
        listX.append(np.random.normal(0, 1, d))
    X = np.array(listX).transpose()
    
    y = np.ones(n)
    p1 = 1
    p2 = 1
    p3 = -0.5
    
    for i in range(n):
        if( p1*X[0,i]+p2*X[1,i]-p3 > 0 ):
            y[i] = -1
    return (d, n, X, y)

# données linéaire avec du bruit
def generer_lineaire_non_separable():
    d = 2
    n = 2000
    
    listX = []
    for i in range(n):
        listX.append(np.random.normal(0, 1, d))
    X = np.array(listX).transpose()
    
    y = np.ones(n)
    p1 = 1
    p2 = 1
    p3 = 0
    
    for i in range(n):
        if( p1*X[0,i]+p2*X[1,i]-p3 > random.gauss(0,0.3)):
            y[i] = -1
    return (d, n, X, y)

# données linéaire avec du bruit en dimension 30
def generer_lineaire_non_separable_dim30():
    d = 30
    n = 2000
    
    listX = []
    for i in range(n):
        listX.append(np.random.normal(0, 1, d))
    X = np.array(listX).transpose()
    
    y = np.ones(n)
    
    for i in range(n):
        if( (listX[i]).sum() > random.gauss(0,0.3)):
            y[i] = -1
    return (d, n, X, y)

# Données séparable par une parabole
def generer_parabole():
    d = 2
    n = 2000
    
    listX = []
    for i in range(n):
        listX.append(np.random.normal(0, 1, d))
    X = np.array(listX).transpose()
    
    y = np.ones(n)
    p1 = 1
    p2 = 1
    p3 = +0.3
    
    for i in range(n):
        if( p1*X[0,i]**2+p2*X[1,i]-p3 > 0 ):
            y[i] = -1
            
    return (d, n, X, y)

# données réelles 
def data_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data.transpose()
    y = ( data.target * 2 ) - 1
    (d, n) = X.shape
    return (d, n, X, y)

# données réelles normalisées
def data_breast_cancer_normalise():
    from sklearn.datasets import load_breast_cancer
    from sklearn import preprocessing
    data = load_breast_cancer()
    X = (preprocessing.StandardScaler().fit(data.data).transform(data.data)).transpose()
    y = ( data.target * 2 ) - 1
    (d, n) = X.shape
    return (d, n, X, y)

##############################################
####          Conditions d'arrêts         ####
############################################## 

# condition d'arrêt par rapport au nombre d'itérations
def arret_100k(t, t_algo):
    return t>100000

def arret_200k(t, t_algo):
    return t>200000

def arret_1k(t, t_algo):
    return t>1000

def arret_10k(t, t_algo):
    return t>10000

# condition d'arrêt par rapport au temps
def arret_300ms(t, t_algo):
    return t_algo > 0.3

def arret_2s(t, t_algo):
    return t_algo > 2

def arret_4s(t, t_algo):
    return t_algo > 4

def arret_10s(t, t_algo):
    return t_algo > 10

def arret_60s(t, t_algo):
    return t_algo > 60


##############################################
####        Génération des graphes        ####
############################################## 
    
# Exemple de visualisation du jeu de données
(d, n, X, y) =generer_lineaire() # à modifier le jeu de données 
(alpha_zero, w_zero) = initialisation(n, d)
visualise( X, y, w_zero, 0, n ) 

# Exemple de visualisation des graphes des algorithmes

# Paramètres à modifier 
# choisir le jeu de données  :
(d, n, X, y) = generer_lineaire()
#  choisir la valeur de lambda :
lambda_ = 0.001 
# choisir la condition d'arrêt :
arret = arret_10s  

(alpha_zero, w_zero) = initialisation(n, d)

print('SGD running')
res_SGD_temp = SGD(X, y, w_zero, n, d, lambda_, arret, True, 2000)
print('SDCA running')
res_SDCA_temp = SDCA(X, y, alpha_zero, w_zero, n, d, lambda_, 'Average', arret, True, 2000)
print('SDCA_perm running')
res_SDCA_perm_temp = SDCA_perm(X, y, alpha_zero, w_zero, n, d, lambda_, 'Average', arret, True, 2000)
print('pegasos running')
res_pegasos_temp = pegasos(X, y, w_zero, n, d, lambda_, arret, True, 10000)
print('pegasos_batch running')
res_pegasos_batch_temp = pegasos_batch(X, y, w_zero, n, d, lambda_, arret, True, 2000)

print("erreur de classification : ")
pyplot.plot( res_SGD_temp['label_time'], res_SGD_temp['class_err'], label='SGD', color='orange')
pyplot.plot( res_SDCA_temp['label_time'], res_SDCA_temp['class_err'], label='SDCA', color='red')
pyplot.plot( res_SDCA_perm_temp['label_time'], res_SDCA_perm_temp['class_err'], label='SDCA-Perm', color='green')
pyplot.plot( res_pegasos_temp['label_time'], res_pegasos_temp['class_err'], label='Pegasos', color='grey')
pyplot.plot( res_pegasos_batch_temp['label_time'], res_pegasos_batch_temp['class_err'], label='Pegasos Mini-Batch', color='blue')

pyplot.legend()
pyplot.ylim(0, 0.250)
pyplot.show()


print("erreur du probleme de minimisation : ")
pyplot.plot( res_SGD_temp['label_time'], res_SGD_temp['min_problem'], label='SGD', color='orange')
pyplot.plot( res_SDCA_temp['label_time'], res_SDCA_temp['min_problem'], label='SDCA', color='red')
pyplot.plot( res_SDCA_perm_temp['label_time'], res_SDCA_perm_temp['min_problem'], label='SDCA-Perm', color='green')
pyplot.plot( res_pegasos_temp['label_time'], res_pegasos_temp['min_problem'], label='Pegasos', color='grey')
pyplot.plot( res_pegasos_batch_temp['label_time'], res_pegasos_batch_temp['min_problem'], label='Pegasos Mini-Batch', color='blue')
#pyplot.yscale('log')
pyplot.legend()
pyplot.ylim(0, 0.6)
pyplot.show()

# on affiche l'hyperplan calculé par l'algorithme Pegasos Mini-Batch :
# ( uniquement en dimension 2 )
if( d==2):
    w = res_pegasos_batch_temp['w']
    b = compute_b(X, y, w, n, d)
    visualise( X, y, w, b, n ) 
    