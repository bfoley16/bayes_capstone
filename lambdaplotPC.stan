functions{
  real student_t_pcprior_lpdf(real nu, real lambda) {
    return  log(lambda) + log((nu*(nu+2)*(2*nu + 9) + 4)/(3*nu^2 * (nu+1)^2 * (nu-2))) + lambda*log(nu^2/((nu+1)*(nu-2))) - lambda*((nu+2)/(3*nu * (nu+1)));
  }
}

data {
  int<lower=1> N;
  vector[N] y;
  real<lower=0> lambda;
}


parameters {
  real<lower=0> nu;
}


model {
  //Priors
  target += student_t_pcprior_lpdf(nu | lambda);
  
  //likelihood
  target += student_t_lpdf(y | nu, 0, 1);
}

