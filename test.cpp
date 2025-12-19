#include <omp.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define N 10000000

int main(int argc, char** argv) {
  // omp_set_num_threads(4);
  
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  double* a = new double[N];

  // #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    a[i] = std::sin(static_cast<double>(i));
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  
  return 0;
}