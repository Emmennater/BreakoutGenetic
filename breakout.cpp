#include <iostream>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <omp.h>
#include <evolution.h>
using namespace std;
#define N 1000

int main() {
  omp_set_num_threads(4);

  Evolution evolution(N, 100);
  evolution.train();

  return 0;
}
