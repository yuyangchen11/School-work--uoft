

Comp_priors <- function(train_labels) {
  #' Compute the priors of each class label 
  #' 
  #' @param train_labels a vector of labels with length equal to n
  #' @param K the number of classes in the response
  #' 
  #' @return a probability vector of length K
  
  K <- length(unique(train_labels))
  pi_vec <- rep(0, K)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  labels <- sort(unique(train_labels)) 
  for (k in 1:length(labels)) {
    pi_vec[k] <- mean(train_labels == labels[k])
  }
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(pi_vec)
}
  


Comp_cond_means <- function(train_data, train_labels) {
  #' Compute the conditional means of each class 
  #' 
  #' @param train_data a n by p matrix containing p features of n training points
  #' @param train_labels a vector of labels with length equal to n
  #' @param K the number of levels in the response
  #' 
  #' @return a p by K matrix, each column represents the conditional mean given
  #'   each class.
  
  K <- length(unique(train_labels))
  p <- ncol(train_data)
  mean_mat <- matrix(0, p, K)
  labels <- sort(unique(train_labels))
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  for (k in 1:K) { 
    nk <- sum(train_labels==labels[k])
    x <- rep(0, p)
    for (i in which(train_labels==labels[k])) {
      x <- x + train_data[i, ]
    }
    mean_mat[,k] <- (1/nk) * x
  }
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(mean_mat)
}



Comp_cond_covs <- function(train_data, train_labels, K, method) {
  #' Compute the conditional covariance matrix of each class
  #' 
  #' @param train_data a n by p matrix containing p features of n training points
  #' @param train_labels a vector of labels with length equal to n
  #' @param K the number of levels in the response
  #' @param method one of the methods in "LDA", "QDA" and "NB"
  #' 
  #' @return 
  #'  if \code{method} is "QDA", return an array with dimension (p, p, K),
  #'    containing p by p covariance matrices of each class;
  #'  else if \code{method} is "NB", return a p by K matrix containing the 
  #'    diagonal covariance entries of each class; 
  #'  else return a p by p covariance matrix.
  
  p <- ncol(train_data)
  N <- nrow(train_data)
  mean_mat <- Comp_cond_means(train_data, train_labels)
  labels <- sort(unique(train_labels))
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  
  if (method == "LDA") {
    cov_arr <- matrix(0, p, p)
    for (k in 1:K) {
      cov_arr_temp <- matrix(0, p, p)
      for (i in which(train_labels==labels[k])) {
        cov_arr_temp <- cov_arr_temp + (train_data[i,]-mean_mat[,k]) %*% t(train_data[i,]-mean_mat[,k])
      }
      cov_arr <- cov_arr + cov_arr_temp
    }
    cov_arr <- (1/(N-k))*cov_arr
  } else if (method == "QDA") {
    cov_arr <- array(0, dim=c(p,p,K))
    for (k in 1:K) {
      nk <- sum(train_labels==labels[k])
      cov_arr_temp <- matrix(0, p, p)
      for (i in which(train_labels==labels[k])) {
        cov_arr_temp <- cov_arr_temp + (train_data[i,]-mean_mat[,k]) %*% t(train_data[i,]-mean_mat[,k])
      }
      cov_arr[,,k] <- (1/(nk-1))*cov_arr_temp
    }
  } else {
    cov_arr <- array(0, dim=c(p,p,K))
    for (k in 1:K) {
      nk <- sum(train_labels==labels[k])
      cov_arr_temp <- matrix(0, p, p)
      for (i in which(train_labels==labels[k])) {
        cov_arr_temp <- cov_arr_temp + (train_data[i,]-mean_mat[,k]) %*% t(train_data[i,]-mean_mat[,k])
      }
      cov_arr[,,k] <- (1/(nk-1))*cov_arr_temp
      cov_arr[,,k] <- diag(diag(cov_arr[,,k]))
    }
  }
  
  return(cov_arr)
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
}




Predict_posterior <- function(test_data, priors, means, covs, method) {
  
  #' Predict the posterior probabilities of each class 
  #'
  #' @param test_data a n_test by p feature matrix 
  #' @param priors a vector of prior probabilities with length equal to K
  #' @param means a p by K matrix containing conditional means given each class
  #' @param covs covariance matrices of each class, depending on \code{method}
  #' @param method one of the methods in "LDA", "QDA" and "NB"
  #'   
  #' @return a n_test by K matrix: each row contains the posterior probabilities 
  #'   of each class.
  
  n_test <- nrow(test_data)
  K <- length(priors)
  
  posteriors <- matrix(0, n_test, K) 
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  library(mvtnorm)
  if (method == "LDA") {
    denom <- numeric(n_test)
    for (i in 1:n_test) {
      a <- 0
      for (k in 1:K) {
        a <- a + priors[k]*dmvnorm(x=test_data[i,], mean=means[,k], sigma=covs)  
      }
      if (a == 0) {a <- 1e-10}
      denom[i] <- a
    } 
    for (i in 1:n_test) {
      for (k in 1:K) {
        posteriors[i,k] <- priors[k]*dmvnorm(x=test_data[i,], mean=means[,k], sigma=covs) / denom[i]
      }
    }
  } else {
    denom <- numeric(n_test)
    for (i in 1:n_test) {
      a <- 0
      for (k in 1:K) {
        a <- a + priors[k]*dmvnorm(x=test_data[i,], mean=means[,k], sigma=covs[,,k])  
      }
      if (a == 0) {a <- 1e-10}
      denom[i] <- a
    } 
    for (i in 1:n_test) {
      for (k in 1:K) {
        posteriors[i,k] <- priors[k]*dmvnorm(x=test_data[i,], mean=means[,k], sigma=covs[,,k]) / denom[i]
      }
    }
  }
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(posteriors)
}


Predict_labels <- function(posteriors) {
  
  #' Predict labels based on the posterior probabilities over K classes
  #' 
  #' @param posteriors A n by K posterior probabilities
  #' 
  #' @return A vector of predicted labels with length equal to n
  
  n_test <- nrow(posteriors)
  pred_labels <- rep(NA, n_test)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  pred_labels <- apply(posteriors, 1, which.max) - 1
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(pred_labels)
}




