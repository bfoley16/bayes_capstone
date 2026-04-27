data {
  int<lower=0> N;
  vector[N] x; // avg_points
  vector[N] mu;
  matrix[N, N] R;

  real<lower=0> a;
  real<lower=0> b;
}

transformed data {
  // log observations
  vector[N] log_x;
  log_x = log(x);
  
  // normalize observations
  vector[N] norm_x;
  norm_x = log_x - mean(log_x);
}

parameters {
  real<lower=0> tau;
}

model {
  // likelihood
  target += multi_normal_prec_lpdf(norm_x | mu, tau * R);

  // gamma prior on tau
  target += gamma_lpdf(tau|a,b);
}

generated quantities {
  vector[N] x_pred = multi_normal_rng(mu, ((1/tau)^(0.5) * inverse(R))); 
}
