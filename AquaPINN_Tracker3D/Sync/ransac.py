# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def ransac(X, y, n_samples=10, max_iterations=100, threshold=1):
   best_model = None
   best_inliers_count = 0
   best_error = float('inf')
   best_errors=np.zeros(max_iterations)
   best_inliers_counts=np.zeros(max_iterations)
   for ii in range(max_iterations):
       # 随机选择子集
       sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
       X_sample, y_sample = X[sample_indices], y[sample_indices]
       # 拟合线性模型
       model = LinearRegression().fit(X_sample, y_sample)
       y_pred = model.predict(X)
       # 计算误差和内点数
       residuals = np.abs(y - y_pred)
       inliers = residuals < threshold
       inliers_count = np.sum(inliers)
       if inliers_count>1:
           error = mean_squared_error(y[inliers], y_pred[inliers])
           # 更新最佳模型
           if inliers_count > best_inliers_count or (inliers_count == best_inliers_count and error < best_error):
               best_inliers_count = inliers_count
               best_model = model
               best_error = error
       best_errors[ii] = best_error
       best_inliers_counts[ii]=best_inliers_count
   return best_model, best_inliers_counts, best_errors
