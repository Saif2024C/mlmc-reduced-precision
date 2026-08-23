/*
   P = mlmc(Lmin,Lmax,N0,eps, mlmc_l, alpha,beta,gamma, Nl,Cl)
 
   multilevel Monte Carlo control routine

   Lmin  = minimum level of refinement       >= 2
   Lmax  = maximum level of refinement       >= Lmin
   N0    = initial number of samples         > 0
   eps   = desired accuracy (rms error)      > 0 
 
   mlmc_l(l,N,sums)   low-level function
        l       = level
        N       = number of paths
        sums[0] = sum(cost)
        sums[1] = sum(Y)
        sums[2] = sum(Y^2)
        where Y are iid samples with expected value:
        E[P_0]           on level 0
        E[P_l - P_{l-1}] on level l>0

   alpha -> weak error is  O(2^{-alpha*l})
   beta  -> variance is    O(2^{-beta*l})
   gamma -> sample cost is O(2^{gamma*l})

   if alpha, beta, gamma are not positive then they will be estimated

   P   = value
   Nl  = number of samples at each level
   Cl  = average cost of samples at each level
*/


#ifndef NESTED_MLMC_CPP_INCLUDED
#define NESTED_MLMC_CPP_INCLUDED

#include <math.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>


void regression(int, float *, float *, float &a, float &b);

float mlmc(int Lmin, int Lmax, int N0, float eps,
           void (*mlmc_l)(int, int, double *),
           float alpha_0, float beta_0, float gamma_0,
           int *Nl, float *Cl) {

  double sums[7], suml[3][42];
  float  ml[42], Vl[42], NlCl[42], x[42], y[42],
         alpha, beta, gamma, sum, theta;
  int    dNl[42], L, converged;

  int    diag = 0;  // diagnostics, set to 0 for none 

  //
  // check input parameters
  //

  if (Lmin<2) {
    fprintf(stderr,"error: needs Lmin >= 2 \n");
    exit(1);
  }
  if (Lmax<Lmin) {
    fprintf(stderr,"error: needs Lmax >= Lmin \n");
    exit(1);
  }

  if (N0<=0 || eps<=0.0f) {
    fprintf(stderr,"error: needs N>0, eps>0 \n");
    exit(1);
  }

  //
  // initialisation
  //

  alpha = fmax(0.0f,alpha_0);
  beta  = fmax(0.0f,beta_0);
  gamma = fmax(0.0f,gamma_0);
  theta = 0.25f;             // MSE split between bias^2 and variance

  L = Lmin;
  converged = 0;

  for(int l=0; l<=2*Lmax+1; l++) {
    Nl[l]   = 0;
    Cl[l]   = powf(2.0f,(float)l*gamma);
    NlCl[l] = 0.0f;

    for(int n=0; n<3; n++) suml[n][l] = 0.0;
  }

  /*
===========================================================================
NESTED MLMC LEVEL INDEXING
===========================================================================

Standard MLMC uses one hierarchy of timestep refinements:

    Level ℓ = 0 : P0
    Level ℓ = 1 : P1 - P0
    Level ℓ = 2 : P2 - P1
    ...

where

    Pℓ = payoff computed using timestep hℓ.

In the nested MLMC formulation we introduce TWO versions of each level:

    Pℓ      = high precision / exact calculation
    Peℓ     = low precision / approximate calculation

and split each MLMC correction into

    (Pℓ - Pℓ-1)

        =

    (Pe_fℓ - Pe_cℓ-1)

        +

    [ (Pℓ - Pℓ-1)
      - (Pe_fℓ - Pe_cℓ-1) ]

The first term is cheap because it uses low precision arithmetic or
approximate random numbers.

The second term is a correction which removes the bias introduced by
the cheap approximation.

Since

    cheap term + correction term = exact MLMC correction

the telescoping sum remains unchanged.

---------------------------------------------------------------------------
INDEX MAPPING
---------------------------------------------------------------------------

The nested MLMC driver doubles the number of levels.

Physical timestep level ℓ:

    ℓ = 0
    ℓ = 1
    ℓ = 2
    ...

is represented internally by MLMC level index l:

    l = 2ℓ     -> cheap estimator
    l = 2ℓ + 1 -> correction estimator

Example:

    l = 0  : cheap level ℓ = 0
    l = 1  : correction level ℓ = 0

    l = 2  : cheap level ℓ = 1
    l = 3  : correction level ℓ = 1

    l = 4  : cheap level ℓ = 2
    l = 5  : correction level ℓ = 2

etc.

Therefore an original MLMC simulation with levels

    ℓ = 0,...,L

becomes

    l = 0,...,2L+1

in the nested MLMC implementation.

---------------------------------------------------------------------------
EVEN LEVELS
---------------------------------------------------------------------------

For l = 2ℓ:

    Y_l = Pe_fℓ - Pe_cℓ-1

This is the low precision MLMC correction.

Most samples are usually taken here because the cost is low.

---------------------------------------------------------------------------
ODD LEVELS
---------------------------------------------------------------------------

For l = 2ℓ+1:

    Y_l = (Pℓ - Pℓ-1)
          - (Pe_fℓ - Pe_cℓ-1)

This is the correction to the low precision estimate.

The variance is usually small because

    Pℓ ≈ Peℓ

so only a relatively small number of expensive samples are required.

---------------------------------------------------------------------------
KEY IDEA
---------------------------------------------------------------------------

Standard MLMC reduces cost by using many coarse timestep simulations.

Nested MLMC reduces cost further by also using many low precision /
approximate simulations and only a few expensive correction samples.

===========================================================================
*/

  for(int l=0; l<=2*L+1; l++) dNl[l] = N0;

  //
  // main loop
  //

  while (!converged) {

    //
    // update sample sums
    //

    if (diag) {
      for (int l=0; l<=2*L+1; l++) printf(" %d ",dNl[l]);
      printf(" \n");
    }
    
    for (int l=0; l<=2*L+1; l++) {
      if (dNl[l]>0) {
        for(int n=0; n<7; n++) sums[n] = 0.0;
        mlmc_l(l,dNl[l],sums);
        suml[0][l] += (float) dNl[l];
        suml[1][l] += sums[1];
        suml[2][l] += sums[2];
        NlCl[l]    += sums[0];  // sum total cost
      }
    }

    //
    // compute absolute average, variance and cost,
    // correct for possible under-sampling,
    // and set optimal number of new samples
    //

    sum = 0.0f;
    
    // sum[2][l] = sigma Y_i^2
    // sum[1][l] = sigma Y_i
    // sum[0]l] = N (sample count)

    // standard FP16 (half precision) saturates at 65,504
    // The smallest non-zero positive number (subnormal) is approximately 
    // 5.96 × 10⁻⁸. Any number smaller than this will underflow to zero
    
    for (int l=0; l<=2*L+1; l++) {
      ml[l] = fabs(suml[1][l]/suml[0][l]); //sample mean. 
      // E[Y]^2 - E[Y^2]
      Vl[l] = fmaxf(suml[2][l]/suml[0][l] - ml[l]*ml[l], 0); 
      if (gamma_0 <= 0.0f) Cl[l] = NlCl[l] / suml[0][l];

      if (l>3) {
        ml[l] = fmaxf(ml[l],  0.5f*ml[l-2]/powf(2.0f,alpha));
        Vl[l] = fmaxf(Vl[l],  0.5f*Vl[l-2]/powf(2.0f,beta));
      }

      sum += sqrtf(Vl[l]*Cl[l]);
    }

    for (int l=0; l<=2*L+1; l++) {
      dNl[l] = ceilf( fmaxf( 0.0f, 
                       sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                     - suml[0][l] ) );
    }
    
 
    //
    // use linear regression to estimate alpha, beta, gamma if not given
    //

    if (alpha_0 <= 0.0f) {
      for (int l=1; l<=L; l++) {
        x[l-1] = l;
        y[l-1] = - log2f(ml[2*l]);
      }
      regression(L,x,y,alpha,sum);
      alpha = fmax(alpha,0.5f);
      
      if (diag) printf(" alpha = %f \n",alpha);
    }

    if (beta_0 <= 0.0f) {
      for (int l=1; l<=L; l++) {
        x[l-1] = l;
        y[l-1] = - log2f(Vl[2*l]);
      }
      regression(L,x,y,beta,sum);
      beta = fmax(beta,0.5f);

      if (diag) printf(" beta = %f \n",beta);
    }

     if (gamma_0 <= 0.0f) {
      for (int l=1; l<=L; l++) {
        x[l-1] = l;
        y[l-1] = log2f(Cl[2*l]);
      }
      regression(L,x,y,gamma,sum);
      gamma = fmax(gamma,0.5f);

      if (diag) printf(" gamma = %f \n",gamma);
    }

    //
    // if (almost) converged, estimate remaining error and decide 
    // whether a new level is required
    //

    sum = 0.0;
      for (int l=0; l<=2*L+1; l++)
        sum += fmaxf(0.0f, (float)dNl[l]-0.01f*suml[0][l]);

    if (sum==0) {
      if (diag) printf(" achieved variance target \n");

      converged = 1;
      float rem = ml[2*L] / (powf(2.0f,alpha)-1.0f);

      if (rem > sqrtf(theta)*eps) {
        if (L==Lmax)
          printf("*** failed to achieve weak convergence *** \n");
        else {
          converged = 0;
          L++;
          Vl[2*L]   = Vl[2*L-2]/powf(2.0f,beta);
          Cl[2*L]   = Cl[2*L-2]*powf(2.0f,gamma);
          Vl[2*L+1] = Vl[2*L-1]/powf(2.0f,beta);
          Cl[2*L+1] = Cl[2*L-1]*powf(2.0f,gamma);

          if (diag) printf(" L = %d \n",L);

          sum = 0.0f;
          for (int l=0; l<=2*L+1; l++) sum += sqrtf(Vl[l]*Cl[l]);
          for (int l=0; l<=2*L+1; l++)
            dNl[l] = ceilf( fmaxf( 0.0f, 
                            sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                          - suml[0][l] ) );
        }
      }
    }
  }

  //
  // finally, evaluate multilevel estimator and set outputs
  //

  float P = 0.0f;
  for (int l=0; l<=2*L+1; l++) {
    P    += suml[1][l]/suml[0][l];
    Nl[l] = suml[0][l];
    Cl[l] = NlCl[l] / Nl[l];
  }

  return P;
}


//
// linear regression routine
//

void regression(int N, float *x, float *y, float &a, float &b){

  float sum0=0.0f, sum1=0.0f, sum2=0.0f, sumy0=0.0f, sumy1=0.0f;

  for (int i=0; i<N; i++) {
    sum0  += 1.0f;
    sum1  += x[i];
    sum2  += x[i]*x[i];

    sumy0 += y[i];
    sumy1 += y[i]*x[i];
  }

  a = (sum0*sumy1 - sum1*sumy0) / (sum0*sum2 - sum1*sum1);
  b = (sum2*sumy0 - sum1*sumy1) / (sum0*sum2 - sum1*sum1);
}
#endif // NESTED_MLMC_CPP_INCLUDED
