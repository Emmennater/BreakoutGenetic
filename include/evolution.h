#include <net.h>

class Evolution {
  mt19937 rng = mt19937(time(0));
  uniform_real_distribution<float> dist = uniform_real_distribution<float>(0.0f, 2.0f);
  
  const float cross_rate = 0.0f;
  const int top_elite = 5;
  const int top_survive = 20;
  const int init_lives = 1;
  const int max_lives = 5;
  const int log_every = 100;
  const int save_every = 100;
  const int max_steps = 10000;
  const int rollouts = 1;
  const int step_size = 100;
  const int pop_size;
  const int training_steps;
  int training_step = 0;
  vector<State> states;
  vector<Genome> genomes;
  vector<float> fitness;
  vector<int> dones;
  vector<int> lives;
  vector<size_t> sorted_idxs;
  
  int chooseElite(int j);
  void resetStates();
  void resetFitness();
  void step();
  bool genDone();
  void removeLives();
  void evaluate();
  void nextGen();
  void logFitness();
  void sortFitness();
  void save();
  void load();
public:
  Evolution(int pop_size, int training_steps);
  void train();
};
