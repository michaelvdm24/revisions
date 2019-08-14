# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 21:33:55 2018

@author: micha
"""
import numpy as np 
import matplotlib.pyplot as plt 
import time 

def my_training(train_cat, train_grass):
    train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter = ','))
    train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter = ','))
    mu_cat = np.mean(train_cat,1)
    Sigma_cat = np.cov(train_cat)
    mu_grass = np.mean(train_grass,1)
    Sigma_grass = np.cov(train_grass)
    
    return mu_cat, mu_grass, Sigma_cat, Sigma_grass

def my_testing(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass): 
    coeff_cat = 1/(((2*np.pi)**32)*(np.linalg.det(Sigma_cat)**(1/2)))
    coeff_grass = 1/(((2*np.pi)**32)*(np.linalg.det(Sigma_grass)**(1/2))) #Calculates both coefficients for multivariate gaussian
    inv_cat = np.linalg.pinv(Sigma_cat)
    inv_grass = np.linalg.pinv(Sigma_grass) #inverse Sigma 
    prior_cat = K_cat / (K_cat + K_grass) #Create prior distributions or Fclass constants
    prior_grass = K_grass / (K_cat + K_grass)
    Output = np.zeros((375-8,500-8)) #Makes empty array which is same dimensions as cat image but of zeros
    for i in range(375-8): #cols
        for j in range(500-8): #rows
            z = Y[i:i+8, j:j+8] #Create 8x8 patch
            z_vector = z.flatten('F') #Convert 8x8 patch into a 64x1 column vector
            z = np.reshape(z_vector, (64, 1), order = 'F')  #Shapes the 64 x 64 array into a 
            f_cat_conditional = np.dot(np.dot((-(0.5)*np.transpose(z - mu_cat)),inv_cat),(z - mu_cat)) 
            f_grass_conditional = np.dot(np.dot((-(0.5)*np.transpose(z - mu_grass)),inv_grass),(z - mu_grass)) 
            f_cat = coeff_cat * np.exp(f_cat_conditional)
            f_grass = coeff_grass * np.exp(f_grass_conditional)
            f_cat_posterior = np.dot(f_cat,prior_cat)   #Probability that we see the cat class label given a patch Z 
            f_grass_posterior = np.dot(f_grass,prior_grass)     #Probability that we see the grass class label given a patch Z
            if (f_cat_posterior * 0.0001) >= (f_grass_posterior): 
                Output[i,j] = 1
    return Output

train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter = ','))
Y = plt.imread('cat_grass.jpg')/255
K_cat = train_cat.shape[1] #.shape[1] gives the number of columns in the matrix
K_grass = train_grass.shape[1]

mu_cat, mu_grass, Sigma_cat, Sigma_grass = my_training(train_cat, train_grass)
start_time = time.time()
Image = my_testing(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass)
time_run = time.time() - start_time
plt.imshow(Image*255, cmap = 'gray')
print('My runtime is %s seconds' % time_run)
Truth = plt.imread('truth.png')
ground = Truth[0:375-8,0:500-8] #Dimensions, N, are 367 x 492
error = np.sum(np.abs(Image - ground))/(367*492) #Mean absolute error calculation between image and ground truth
print('the MAE is %s' %error)
