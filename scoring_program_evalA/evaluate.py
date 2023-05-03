#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd 
import os
import os.path


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def precision(tp, fp):
  try:
    return tp / float(fp + tp)
  except ZeroDivisionError:
    return 0.0
def recall(tp, fn):
  try:
    return tp / float(fn + tp)
  except ZeroDivisionError:
    return 0.0

def f1(actual, predicted, label):

    """ A helper function to calculate f1-score for the given `label` """

    # F1 = 2 * (precision * recall) / (precision + recall)

    tp,fp,tn,fn=perf_measure(actual, predicted)
    
    pre = precision(tp,fp)
    re = recall(tp,fn)
    f1 = 2 * (pre * re) / (pre + re)
    return f1

def f1_macro(actual, predicted):
    # `macro` f1- unweighted mean of f1 per label
    return np.mean([f1(actual, predicted, label) 
        for label in np.unique(actual)])



input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir,'res')
truth_dir = os.path.join(input_dir,'ref')

if not os.path.isdir(submit_dir):
  print("%s doesn't exist" %submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  output_filename = os.path.join(output_dir,'scores.txt')
  output_file = open(output_filename,'wb')

  truth_file = os.path.join(truth_dir,"evalA_ref.csv")

  submission_answer_file = os.path.join(submit_dir,"predictions.csv")

  truth = pd.read_csv(truth_file)
  pred = pd.read_csv(submission_answer_file)

  merged = truth.merge(pred, on = 'index',suffixes =('_true','_pred'),how='left')

  result = f1_macro(merged['label_true'].values,merged['label_pred'].values)

  if(np.isnan(result)):
    result = 404
  output_file.write("f1-score:" + str(result))

  output_file.close()