#include <Eigen/Core>
#include <iostream>
#include <unsupported/Eigen/SpecialFunctions>
using namespace Eigen;
int main() {
  Array4d v(-0.5, 2, 0, -7);
  std::cout << v.erf() << std::endl;
}
