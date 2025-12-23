#include <evolution.h>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <argsort.h>
namespace fs = std::filesystem;

Evolution::Evolution(int pop_size, int training_steps)
  : pop_size(pop_size), training_steps(training_steps) {
  states.resize(pop_size);
  genomes.resize(pop_size);
  fitness.resize(pop_size);
  dones.resize(pop_size);
  lives.resize(pop_size);

  for (int i = 0; i < pop_size; i++) {
    genomes[i].init(rng);
    lives[i] = init_lives;
  }

  load();
}

void Evolution::resetGen() {
  for (int i = 0; i < pop_size; i++) {
    states[i] = State();
    dones[i] = 0;
    fitness[i] = 0;
  }
}

void Evolution::step() {
  constexpr int step_size = 100;
  static float randoms[step_size];

  for (int i = 0; i < step_size; i++) {
    randoms[i] = dist(rng);
  }

  #pragma omp parallel for
  for (int i = 0; i < pop_size; i++) {
    for (int j = 0; j < step_size; j++) {
      if (!dones[i]) {
        int action = forward(genomes[i], states[i]);
        envStep(&states[i], &action, &fitness[i], &dones[i], randoms[j]);
      }
    }
  }
}

bool Evolution::genDone() {
  for (int i = 0; i < pop_size; i++) {
    if (!dones[i]) {
      return false;
    }
  }
  return true;
}

void Evolution::evaluate() {
  constexpr int step_size = 100;
  for (int i = 0; i < max_steps / step_size; i++) {
    if (genDone()) break;
    step();
  }

  training_step++;
}

int Evolution::chooseElite() {
  // Sample with probability proportional to fitness
  // Each fitness is already greater than 0.
  const float pwr = 1.0f;
  float total = 0;
  
  for (int i = 0; i < pop_size; i++) {
    total += pow(fitness[i], pwr);
  }
  
  float r = dist(rng) * total;
  
  for (int i = 0; i < pop_size; i++) {
    r -= pow(fitness[i], pwr);
    if (r <= 0) return i;
  }
  
  return pop_size - 1;
}

void Evolution::removeLives() {
  // Find the nth highest fitness
  const int n = pop_size * top_percent;
  float nth = fitness[sorted_idxs[pop_size - n - 1]];
  
  // Remove lives of genomes with fitness below nth
  // Add lives of genomes with fitness above or equal to nth
  for (int i = 0; i < pop_size; i++) {
    int diff = (fitness[i] >= nth) * 2 - 1;
    lives[i] = min(lives[i] + diff, max_lives);
  }
}

void Evolution::nextGen() {
  int j = -1;
  
  for (int i = 0; i < pop_size; i++) {
    // 1) Find two children to replace
    if (lives[i] > 0) continue;
    if (j == -1) { j = i; continue; }

    // 2) Choose elites (could be the same genome)
    int elite1 = chooseElite();
    int elite2 = chooseElite();

    // 3) Copy elites
    Genome child1(genomes[elite1]);
    Genome child2(genomes[elite2]);

    // 4) Crossover
    crossover(child1, child2, rng);

    // 5) Mutate
    mutate(child1, rng);
    mutate(child2, rng);

    // 6) Replace
    genomes[i] = child1;
    genomes[j] = child2;
    lives[i] = init_lives;
    lives[j] = init_lives;
    
    // 7) Reset
    j = -1;
  }

  // Odd number of dead genomes
  if (j != -1) {
    int elite = chooseElite();
    Genome child(genomes[elite]);
    mutate(child, rng);
    genomes[j] = child;
    lives[j] = init_lives;
  }
}

void Evolution::train() {
  cout << "Training for " << training_steps << " steps..." << endl;
  for (int i = 0; i < training_steps; i++) {
    resetGen();
    evaluate();
    sortFitness();
    if (training_step % log_every == 0) logFitness();
    if (training_step % save_every == 0) save();
    removeLives();
    nextGen();
  }
}

void Evolution::logFitness() {
  int n = min(pop_size, 10);

  cout << "Step " << setw(5) << training_step << ": " << fixed << setprecision(2);
  
  for (int i = 0; i < n; i++) {
    size_t j = sorted_idxs[pop_size - i - 1];
    cout << " " << setw(6) << fitness[j];
  }

  cout << endl;
}

void Evolution::sortFitness() {
  sorted_idxs = argsort(fitness);
}

void Evolution::save() {
  if (!fs::exists("genomes")) fs::create_directory("genomes");

  for (int i = 0; i < 10; i++) {
    size_t j = sorted_idxs[pop_size - i - 1];
    saveGenome(genomes[j], "genomes/" + to_string(i) + ".net");
  }

  std::ofstream stats("genomes/stats.txt");
  stats << training_step << endl;
  stats.close();
}

void Evolution::load() {
  if (!fs::exists("genomes")) return;

  for (int i = 0; i < 10; i++) {
    loadGenome(genomes[i], "genomes/" + to_string(i) + ".net");
  }

  std::ifstream stats("genomes/stats.txt");
  stats >> training_step;
  stats.close();
}
