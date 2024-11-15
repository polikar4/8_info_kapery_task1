import numpy as np
import pyqiopt as pq
import pandas as pd
import math

count_decemall = 10 + 1
max_iriska = 0.2
file_output = "task1/output.txt"
file_data = "task1/dataSet.csv"
procentil = np.linspace(0, 1, count_decemall)
average = 1

def getDatas(path):
    df = pd.read_csv(path)  
    df = df.replace(",", ".", regex=True).astype(float)
    matrix = df.values
    l = len(matrix)
    profit = []
    for i in range(l):
        profit.append((matrix[l-1,i] - matrix[0,i])/matrix[0,i]) 
    riski = []
    for i in range(l):
        tempy = []
        for j in range(l-1):
            tempy.append((matrix[j+1,i]-matrix[j,i])/matrix[j,i])
        sr = sum(tempy)/len(tempy)
        sumary = 0
        for k in range(l-1):
            sumary += math.pow(tempy[k]-sr,2)/(l-2)
        riski.append(math.sqrt(l*sumary))
    new_profit = []
    new_riski = []
    for i in range(len(profit)):
        if(profit[i]<0):
            new_profit.append(profit[i])
            new_riski.append(riski[i])
    return new_profit, new_riski



iriski = []
dohodnost = [] 
size = 0

def based():
  # based 
  Q = np.zeros((size,size))
  for i in range(size):
      num_sqer = i//count_decemall
      num_count = i%count_decemall
      Q[i][i] = -dohodnost[num_sqer] * procentil[num_count]

  return Q


def by_two_any():
# by two any 
  Q_by_two_any = np.zeros((size,size))
  
  for x in range(size):
      for y in range(size):
          num_count_x = procentil[x%count_decemall]
          num_count_y = procentil[y%count_decemall]
          
          num_sqer_x = x//count_decemall
          num_sqer_y = y//count_decemall
          
          if(num_sqer_x != num_sqer_y):
              Q_by_two_any[x][y] = - (dohodnost[num_sqer_x] * num_count_x + dohodnost[num_sqer_y] * num_count_y)

  return Q_by_two_any

def shtrach_many(P_shtrach_many):
  # many target 
  Q_shtrach_many = np.zeros((size,size))
  
  for x in range(size):
      for y in range(size):
          num_count_x = procentil[x%count_decemall]
          num_count_y = procentil[y%count_decemall]
          
          num_sqer_x = x//count_decemall
          num_sqer_y = y//count_decemall
          
          if(num_count_x +  num_count_y != 1 and num_sqer_x != num_sqer_y):
              Q_shtrach_many[x][y] = P_shtrach_many
  return Q_shtrach_many
            
def iriska(P_iriska):
  # iriska 
  Q_iriska  = np.zeros((size,size))
  
  for x in range(size):
      for y in range(size):
          num_count_x = procentil[x%count_decemall]
          num_count_y = procentil[y%count_decemall]
          
          num_sqer_x = x//count_decemall
          num_sqer_y = y//count_decemall
          
          if(iriski[num_sqer_x] * num_count_x + iriski[num_sqer_y] * num_count_y > max_iriska   and num_sqer_x != num_sqer_y):
              Q_iriska[x][y] = P_iriska
  return Q_iriska

def dabl_by(P_dabl_by):
  #ban to dabl by
  
  Q_dabl_by = np.zeros((size,size))
  
  for x in range(size):
      for y in range(size):
          num_count_x = procentil[x%count_decemall]
          num_count_y = procentil[y%count_decemall]
          
          num_sqer_x = x//count_decemall
          num_sqer_y = y//count_decemall
          
          if(num_sqer_x == num_sqer_y and num_count_x < num_count_y):
              Q_dabl_by[x][y] = P_dabl_by
  return Q_dabl_by
        

def summ_shtrach(Q_iriska, Q_shtrach_many, Q_dabl_by):
  # summ shtrach
  Q_summ_shtrach  = np.zeros((size,size))
  
  for x in range(size):
      for y in range(size):
          Q_summ_shtrach[x][y] = Q_iriska[x][y] + Q_shtrach_many[x][y] + Q_dabl_by[x][y]

  return Q_summ_shtrach
        
        

def final(Q, Q_by_two_any, Q_summ_shtrach):
  # final Q 
  Q_final = np.zeros((size,size))
  
  for x in range(size):
      for y in range(size):
          Q_final[x][y] = Q[x][y] + Q_by_two_any[x][y]
          if(Q_summ_shtrach[x][y] != 0):
              Q_final[x][y] = Q_summ_shtrach[x][y]

  return Q_final
            


if __name__ == "__main__":
  # Заполнение данных 
  iriski, dohodnost = getDatas(file_data)
  size = count_decemall * len(dohodnost)
  #
  Q = based()
  Q_by_two_any = by_two_any()
  #
  Q_shtrach_many = shtrach_many(average)
  Q_iriska = iriska(average)
  Q_dabl_by = dabl_by(average)
  #
  Q_summ_shtrach = summ_shtrach(Q_shtrach_many,Q_iriska,Q_dabl_by)
  #
  Q_final = final(Q, Q_by_two_any, Q_summ_shtrach)
  #
  result = pq.solve(Q_final, number_of_steps=1000, gpu=True, number_of_runs=1)

  #
  s = " ".join(map(str, result.vector))
  with open(file_output, 'w') as file:
    elements = s.strip("[]").split()
    for ii in range(0, len(elements), count_decemall):
        file.write(" ".join(elements[ii:ii+count_decemall]) + '\n')

  