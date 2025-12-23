#include <net.h>
#include <fstream>

Genome::Genome() : data(H * I + H + H * H + H + O * H + O) {}

void Genome::init(mt19937& rng) {
  data.resize(H * I + H + H * H + H + O * H + O);

  uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < data.size(); i++) {
    data[i] = dist(rng);
  }
}

void mutate(Genome& g, mt19937& rng, float rate, float delta) {
  uniform_real_distribution<float> dist_delta(-delta, delta);
  uniform_real_distribution<float> dist_rate(0.0f, 1.0f);

  for (size_t i = 0; i < g.data.size(); i++) {
    g.data[i] += (dist_rate(rng) < rate) * dist_delta(rng);
  }
}

void crossover(Genome& g1, Genome& g2, mt19937& rng, float nc) {
  // Simulated Binary Crossover
  uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (size_t i = 0; i < g1.data.size(); i++) {
    float u = dist(rng);
    float beta = u <= 0.5 ? pow(2 * u, 1.0 / (nc + 1))
      : pow(1 / (2 * (1 - u)), 1 / (nc + 1));
    float x1 = g1.data[i];
    float x2 = g2.data[i];
    g1.data[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2);
    g2.data[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2);
  }
}

Eigen::Matrix<float, I, 1> cast(const State& s) {
  Eigen::Matrix<float, I, 1> x;
  memcpy(x.data(), &s, I * sizeof(float));
  return x;
}

int forward(const Genome& g, const State& s) {
  const Eigen::Matrix<float, I, 1>& x = cast(s);

  using MatInput  = Eigen::Matrix<float, H, I>;
  using VecH      = Eigen::Matrix<float, H, 1>;
  using MatHidden = Eigen::Matrix<float, H, H>;
  using MatOutput = Eigen::Matrix<float, O, H>;
  using VecO      = Eigen::Matrix<float, O, 1>;

  // Extract weights and biases from the flattened genome
  const float* p = g.data.data();
  Eigen::Map<const MatInput>  W1(p); p += H * I;
  Eigen::Map<const VecH>      b1(p); p += H;
  Eigen::Map<const MatHidden> W2(p); p += H * H;
  Eigen::Map<const VecH>      b2(p); p += H;
  Eigen::Map<const MatOutput> W3(p); p += O * H;
  Eigen::Map<const VecO>      b3(p);

  // Forward
  VecH h1 = W1 * x + b1;
  h1 = h1.array().tanh();
  VecH h2 = W2 * h1 + b2;
  h2 = h2.array().tanh();
  Eigen::Matrix<float, O, 1> y = W3 * h2 + b3;

  // Pick largest output
  return y.array().maxCoeff();
}

void saveGenome(const Genome& g, const std::string& filename) {
  std::ofstream out(filename, std::ios::binary);
  out.write(reinterpret_cast<const char*>(g.data.data()), g.data.size() * sizeof(float));
}

void loadGenome(Genome& g, const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read(reinterpret_cast<char*>(g.data.data()), g.data.size() * sizeof(float));
}
