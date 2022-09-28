#include <Eigen/Core>
#include <iostream>
#include <unsupported/Eigen/SpecialFunctions>
using namespace Eigen;
int main() {
  Array4d v(0.5, 10, 0, -1);
  std::cout << v.lgamma() << std::endl;
}
