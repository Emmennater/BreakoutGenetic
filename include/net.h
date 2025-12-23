#pragma once
#include <Eigen/Dense>
#include <random>
#include <env.h>

constexpr int I = STATE_SIZE;
constexpr int H = 16;
constexpr int O = ACTION_SIZE;

struct Genome {
  std::vector<float> data;

  Genome();
  void init(std::mt19937& rng);
};

int forward(const Genome& g, const State& s);

void mutate(Genome& g, mt19937& rng, float rate = 0.05, float delta = 0.05);

void crossover(Genome& g1, Genome& g2, mt19937& rng, float nc = 2.0);

void saveGenome(const Genome& g, const std::string& filename);

void loadGenome(Genome& g, const std::string& filename);
