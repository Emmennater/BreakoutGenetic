#include <iostream>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <omp.h>
#include <evolution.h>
using namespace std;
#define N 50

int main() {
  // omp_set_num_threads(4);
  omp_set_num_threads(omp_get_max_threads() - 1);

  Evolution evolution(N, 10000000);
  evolution.train();

  return 0;
}
