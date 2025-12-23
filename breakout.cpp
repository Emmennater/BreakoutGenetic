#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <env.h>
using namespace std;

int main() {
  omp_set_num_threads(4);

  Environment env;
  cout << env.pos2idx(0.3f, 0.5f) << endl;

  Eigen::MatrixXd m(2, 2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  cout << m << endl;
  
  return 0;
}
