data {
  int<lower=1> N;
  vector[N] y;
  real<lower=0> rate;
}


parameters {
  real<lower=0> nu;
}


model {
  //Priors
  target += exponential_lpdf(nu | rate);
  
  //likelihood
  target += student_t_lpdf(y | nu, 0, 1);
}

