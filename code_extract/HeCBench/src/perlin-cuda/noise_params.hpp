#pragma once

struct NoiseParams {
  float ppu;         // pixels per unit
  int seed;
  int octaves;
  float lacunarity;  // frequency modulation rate per octave
  float persistence; // amplitude modulation rate per octave
};

