data {
  int<lower=0> N;
  vector<lower=0>[N] x; // avg_points
  vector[N] mu;
  matrix[N, N] R;

  real<lower=0> U; // upper bound on precision
  real<lower=0, upper=1> alpha; // weight on U
}

transformed data {
  // log observations
  vector[N] log_x;
  log_x = log(x);
  
  // normalize observations
  vector[N] norm_x;
  norm_x = log_x - mean(log_x);

  // flexibility parameter
  real<lower=0> lambda;
  lambda = -log(alpha) / U;
}

parameters {
  real<lower=0> tau;
}

model {
  // likelihood
  target += multi_normal_prec_lpdf(norm_x | mu, tau * R);

  // PC prior on tau
  target += log(lambda) - 1.5 * log(tau) - lambda / sqrt(tau);
}

generated quantities {
  vector[N] x_pred = multi_normal_rng(mu, ((1/tau)^(0.5) * inverse(R))); 
}
