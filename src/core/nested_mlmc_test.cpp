/*

  mlmc_test(mlmc_l, N,L, N0,Eps,Lmin,Lmax, fp)

  multilevel Monte Carlo test routine

   mlmc_l(l,N,sums)     low-level routine

   inputs:  l = level
            N = number of paths

   output: sums[0] = sum(cost)
           sums[1] = sum(Pf-Pc)
           sums[2] = sum((Pf-Pc).^2)
           sums[3] = sum((Pf-Pc).^3)
           sums[4] = sum((Pf-Pc).^4)
           sums[5] = sum(Pf)
           sums[6] = sum(Pf.^2)

   N      = number of samples for convergence tests
   L      = number of levels for convergence tests

   N0     = initial number of samples
   Eps    = desired accuracy array (terminated by value 0)
   Lmin   = minimum level of refinement
   Lmax   = maximum level of refinement

   fp     = file handle for output
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "nested_mlmc.cpp"

// https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html
// variadic macro to print to both file and stdout
#define PRINTF2(fp, ...) {printf(__VA_ARGS__);fprintf(fp,__VA_ARGS__);}


void mlmc_test(void (*mlmc_l)(int, int, double *), int N,int L,
               int N0, float *Eps, int Lmin, int Lmax, FILE *fp) {

//
// first, convergence tests
//

  // current date/time based on current system
  time_t now = time(NULL);
  char *date = ctime(&now);
  int len = strlen(date);
  date[len-1] = ' ';

  PRINTF2(fp,"\n");
  PRINTF2(fp,"**********************************************************\n");
  PRINTF2(fp,"*** MLMC file version 1.0     produced by              ***\n");
  PRINTF2(fp,"*** C++ mlmc_test on %s         ***\n",date);
  PRINTF2(fp,"**********************************************************\n");
  PRINTF2(fp,"\n");
  PRINTF2(fp,"**********************************************************\n");
  PRINTF2(fp,"*** Convergence tests, kurtosis, telescoping sum check ***\n");
  PRINTF2(fp,"*** using N =%7d samples                           ***\n",N);
  PRINTF2(fp,"**********************************************************\n");
  PRINTF2(fp,"\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)");
  PRINTF2(fp,"    kurtosis     check        cost \n--------------------------");
  PRINTF2(fp,"-------------------------------------------------------------\n");

  double sums[7];
  float *cost = (float *)malloc(2*(L+1)*sizeof(float));
  float *del1 = (float *)malloc(2*(L+1)*sizeof(float));
  float *del2 = (float *)malloc(2*(L+1)*sizeof(float));
  float *var1 = (float *)malloc(2*(L+1)*sizeof(float));
  float *var2 = (float *)malloc(2*(L+1)*sizeof(float));
  float *chk1 = (float *)malloc(2*(L+1)*sizeof(float));
  float *kur1 = (float *)malloc(2*(L+1)*sizeof(float));

  for (int l=0; l<=2*L+1; l++) {
    for (int m=0; m<7; m++) sums[m] = 0.0;

    mlmc_l(l,N,sums);

    for (int m=0; m<7; m++) sums[m] = sums[m]/N;

    cost[l] = sums[0];
    del1[l] = sums[1];
    del2[l] = sums[5];
    var1[l] = fmax(sums[2]-sums[1]*sums[1], 1e-10);
    var2[l] = fmax(sums[6]-sums[5]*sums[5], 1e-10);

    kur1[l]  = (      sums[4]
                - 4.0*sums[3]*sums[1]
                + 6.0*sums[2]*sums[1]*sums[1]
                - 3.0*sums[1]*sums[1]*sums[1]*sums[1] )
             / (var1[l]*var1[l]);

    if (l==1 || (l%2==0))
    // Only odd correction levels greater than 1 get a consistency check.
      chk1[l] = 0.0f;
    else if (cost[l] == 0.0f)
      // A zero-cost odd level carries no precision correction at all: this is
      // the adaptive scheme above its cutoff, where the path is FP32
      // throughout and the scheme is standard non-nested MLMC. 
      chk1[l] = sqrtf((float) N) *
                 fabsf( del1[l-1] + del2[l-3] - del2[l-1] )
        / (3.0f*( sqrtf(var1[l-1]) + sqrtf(var2[l-3]) + sqrtf(var2[l-1]) ));
    else
      chk1[l] = sqrtf((float) N) *
                 fabsf( del1[l-1]  +       del2[l-3]  -       del2[l-1]
                      + del1[l]    +       del2[l-2]  -       del2[l]   )
        / (3.0f*( sqrtf(var1[l-1]) + sqrtf(var2[l-3]) + sqrtf(var2[l-1] )
                + sqrtf(var1[l])   + sqrtf(var2[l-2]) + sqrtf(var2[l]   )));
  
    PRINTF2(fp,"%2d  %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e \n",
    l,del1[l],del2[l],var1[l],var2[l],kur1[l],chk1[l],cost[l]);
  }

//
// print out a warning if kurtosis or consistency check looks bad
//

  if (kur1[2*L] > 100.0f) {
    PRINTF2(fp,"\n WARNING: kurtosis on finest level = %f \n",kur1[2*L]);
    PRINTF2(fp," indicates MLMC correction dominated by a few rare paths; \n");
    PRINTF2(fp," for information on the connection to variance of sample variances,\n");
    PRINTF2(fp," see http://mathworld.wolfram.com/SampleVarianceDistribution.html \n");
  }

  float max_chk = 0.0f;
  for (int l=0; l<=2*L+1; l++) max_chk = fmaxf(max_chk,chk1[l]);
  if (max_chk > 1.0f) {
    PRINTF2(fp,"\n WARNING: maximum consistency error = %f \n",max_chk);
    PRINTF2(fp," indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied \n");
  }

//
// use linear regression to estimate alpha, beta, gamma
//

  float alpha, beta, gamma, foo;
  float *x = (float *)malloc(L*sizeof(float));
  float *y = (float *)malloc(L*sizeof(float));

  for (int l=1; l<=L; l++) {
    x[l-1] = l;
    y[l-1] = - log2f(fabsf(del1[2*l]));
  } 
  regression(L,x,y,alpha,foo);

  for (int l=1; l<=L; l++) {
    x[l-1] = l;
    y[l-1] = - log2f(var1[2*l]);
  } 
  regression(L,x,y,beta,foo);

  for (int l=1; l<=L; l++) {
    x[l-1] = l;
    y[l-1] = log2f(cost[2*l]);
  } 
  regression(L,x,y,gamma,foo);

  PRINTF2(fp,"\n******************************************************\n");
  PRINTF2(fp,"*** Linear regression estimates of MLMC parameters ***\n");
  PRINTF2(fp,"******************************************************\n");
  PRINTF2(fp,"\n alpha = %f  (exponent for MLMC weak convergence)\n",alpha);
  PRINTF2(fp," beta  = %f  (exponent for MLMC variance) \n",beta);
  PRINTF2(fp," gamma = %f  (exponent for MLMC cost) \n",gamma);

//
// second, mlmc complexity tests
//

  PRINTF2(fp,"\n");
  PRINTF2(fp,"***************************** \n");
  PRINTF2(fp,"*** MLMC complexity tests *** \n");
  PRINTF2(fp,"***************************** \n\n");
  PRINTF2(fp,"  eps       value   mlmc_cost   std_cost  savings     N_l \n");
  PRINTF2(fp,"--------------------------------------------------------- \n");
 
  int i=0;
  int   *Nl = (int *)malloc(2*(Lmax+1)*sizeof(int));
  float *Cl = (float *)malloc(2*(Lmax+1)*sizeof(float));

  while (Eps[i]>0) {
    float eps = Eps[i++];

    float P = mlmc(Lmin,Lmax,N0,eps,mlmc_l, alpha,beta,0.0f, Nl,Cl);

    float std_cost = 0.0f, mlmc_cost = 0.0f, theta=0.25f;

    for (int l=0; l<=2*Lmax+1; l++) {
      if (Nl[l]>0) {
        // printf(" l, Cl, cost = %d  %f  %f \n",l,Cl[l],cost[l]);
	mlmc_cost += Nl[l]*Cl[l];
	if ((l%2)==0) {
          // Cost of one path at this grid level.  Normally that is the odd
          // super-level's cost, which runs both precisions and so carries the
          // full per-path price.  Above the adaptive cutoff the odd level does
          // not exist (the scheme is standard non-nested MLMC there), and the
          // even level's own cost is the per-path price; without this the
          // reference cost would be read as zero.
          if (l<=2*L)
            std_cost = var2[l]*(cost[l+1] > 0.0f ? cost[l+1] : cost[l])
                     / ((1.0f-theta)*eps*eps);
          else
            std_cost = var2[2*L]*(Cl[l+1] > 0.0f ? Cl[l+1] : Cl[l])
                     / ((1.0f-theta)*eps*eps);
	}
      }
    }

    PRINTF2(fp,"%.4f  %.4e  %.3e  %.3e  %7.2f ",
	    eps, P, mlmc_cost, std_cost, std_cost/mlmc_cost);
    for (int l=0; Nl[l]>0 && l<=2*Lmax+1; l++) PRINTF2(fp,"%9d",Nl[l]);
    PRINTF2(fp,"\n");
  }
  PRINTF2(fp,"\n");
}