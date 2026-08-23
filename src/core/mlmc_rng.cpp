
//----- C++11 random number generation when not using OpenMP -------------

#ifndef _OPENMP

#include <random>           // C++11 random number generators
#include <functional>

/* some web references

   https://www.cplusplus.com/reference/random/
   https://stackoverflow.com/questions/14023880/c11-random-numbers-and-stdbind-interact-in-unexpected-way/14023935
   https://stackoverflow.com/questions/20671573/c11-stdgenerate-and-stduniform-real-distribution-called-two-times-gives-st

*/

// declare generator and output distributions

std::default_random_engine rng;
std::uniform_real_distribution<float> uniform(0.0f,1.0f);
std::normal_distribution<float> normal(0.0f,1.0f);
std::exponential_distribution<float> exponential(1.0f);

auto next_uniform     = std::bind(std::ref(uniform),     std::ref(rng));
auto next_normal      = std::bind(std::ref(normal),      std::ref(rng));
auto next_exponential = std::bind(std::ref(exponential), std::ref(rng));

void rng_initialisation() {
    rng.seed(1234);
    uniform.reset();
    normal.reset();
    exponential.reset();
}

void rng_termination() {
}

//------- MKL/VSL random number generation when using OpenMP -----------

#else

#include <mkl.h>
#include <mkl_vsl.h>
#include <memory.h>
#include <omp.h>

/* each OpenMP thread has its own VSL RNG and storage */

#define NRV 16384  // number of random variables
VSLStreamStatePtr stream;
float *uniforms,      *normals,      *exponentials;
int    uniforms_count, normals_count, exponentials_count;
#pragma omp threadprivate(stream, uniforms,uniforms_count, \
        normals,normals_count, exponentials,exponentials_count)

//
// RNG routines
//

void rng_initialisation(){
  int tid = omp_get_thread_num();
  vslNewStream(&stream, VSL_BRNG_MRG32K3A,1337);
  long long skip = ((long long) (tid+1)) << 48;
  vslSkipAheadStream(stream,skip);
  uniforms     = (float *)malloc(NRV*sizeof(float));
  normals      = (float *)malloc(NRV*sizeof(float));
  exponentials = (float *)malloc(NRV*sizeof(float));
  uniforms_count     = 0;  // this means there are no random
  normals_count      = 0;  // numbers in the arrays currently
  exponentials_count = 0;  // 
}

void rng_termination(){
  vslDeleteStream(&stream);
  free(uniforms);
  free(normals);
  free(exponentials);
}

float next_uniform(){
  if (uniforms_count==0) {
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                  stream,NRV,uniforms,0.0f,1.0f);
    uniforms_count = NRV;
  }
  return uniforms[--uniforms_count];
}

float next_normal(){
  if (normals_count==0) {
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
                  stream,NRV,normals,0.0f,1.0f);
    normals_count = NRV;
  }
  return normals[--normals_count];
}

float next_exponential(){
  if (exponentials_count==0) {
    vsRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE,
                     stream,NRV,exponentials,0.0f,1.0f);
    exponentials_count = NRV;
  }
  return exponentials[--exponentials_count];
}

#endif

//----------------------------------------------------------------------
