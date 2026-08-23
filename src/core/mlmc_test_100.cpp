/*

  mlmc_test_100(mlmc_l, val, N0,Eps,Lmin,Lmax, fp, varargin)
 
  test routine to perform 100 independent MLMC calculations in parallel
 
  sums = mlmc_l(l,N, varargin)     low-level routine
 
  inputs:  l = level
           N = number of paths
 
  output: sums(1) = sum(Pf-Pc)
          sums(2) = sum((Pf-Pc).^2)
 
  val      = exact value (NaN if not known)
  N0       = initial number of samples for MLMC calcs
  Eps      = desired accuracy array for MLMC calcs
  Lmin     = minimum number of levels for MLMC calcs
  Lmax     = maximum number of levels for MLMC calcs
  fp       = file handle for printing to file
  varargin = optional additional user variables to be passed to mlmc_l

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

// why not just include mlmc.cpp instead???
#include "mlmc_test.cpp"      // master MLMC file

// https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html
// variadic macro to print to both file and stdout
// prints out on both file terminal and output text file. 
#define PRINTF2(fp, ...) {printf(__VA_ARGS__);fprintf(fp,__VA_ARGS__);}


void mlmc_test_100(void (*mlmc_l)(int, int, double *), float val,
                   int N0, float *Eps, int Lmin, int Lmax, FILE *fp){

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
  PRINTF2(fp,"***************************************** \n");
  PRINTF2(fp,"*** MLMC errors from 100 calculations *** \n");
  PRINTF2(fp,"***************************************** \n");

  if (isnan(val)) {
    PRINTF2(fp,"\n Exact value unknown \n");
  }
  else {
    PRINTF2(fp,"\n Exact value: %f \n",val);
  }

  int   i = 0;
  int   *Nl = (int *)malloc((Lmax+1)*sizeof(int));
  float *Cl = (float *)malloc((Lmax+1)*sizeof(float));

  while (Eps[i]>0) {
    float eps = Eps[i++];
    PRINTF2(fp,"\n eps = %.3e \n-----------------\n",eps); 

    for(int j=0; j<100; j++) { 
      float P = mlmc(Lmin,Lmax,N0,eps,mlmc_l, 0.0f,0.0f,0.0f, Nl,Cl);
      PRINTF2(fp," %.5e ",P);
      if (j%5==4) PRINTF2(fp,"\n");
    }
  }
}
