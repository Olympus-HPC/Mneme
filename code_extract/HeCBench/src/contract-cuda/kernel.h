const int nContractions = 18;  // the device kernel contains 18 cases

template <typename T>
__global__ void contraction (
  const T *__restrict__ tensor,
  const T *__restrict__ adj,
        T *__restrict__ value,
  const int output_size, 
  const int N, 
  const int nChanels)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < output_size) {  
    int C = nChanels;
    int B = N * C;
    int A = N * B;
    int Y = nChanels * nContractions;

    int f = (tid % Y) % nChanels;
    int Case = (tid % Y) / nChanels + 1;
    int y = (tid / Y) % N;
    int x = (tid / Y) / N;

    int a, b, c, d, e;
    T adj_value;

    T sum = (T)0;

    // +-----------+
    // | 1 + 1 + 1 |
    // +-----------+

    // Case 1 (1/50): Fix a, b. Contract c, d, e.
    if (Case == 1) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    // Case 2 (3/50): Fix a, d. Contract b, c, e.
    if (Case == 2) {    
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (b = 0; b < N; ++b) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }  
    }

    // Case 3 (5/50): Fix b, c. Contract a, d, e.
    if (Case == 3) {    
      b = x;
      c = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            for (a = 0; a < N; ++a) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }  
    }

    // Case 4 (6/50): Fix b, d. Contract a, c, e.
    if (Case == 4) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (a = 0; a < N; ++a) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    // Case 5 (10/50): Fix d, e. Contract a, b, c.
    if (Case == 5) {    
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (a = 0; a < N; ++a) {
          for (b = 0; b < N; ++b) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    // +-------+
    // | 1 + 2 |
    // +-------+

    // Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
    if (Case == 6) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          c = d;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }

    // Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
    if (Case == 7) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        e = d;
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
    if (Case == 8) {
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (b = 0; b < N; ++b) {
            c = b;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
    if (Case == 9) {
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          b = e;
          for (c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
    if (Case == 10) {
      b = x;
      c = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            a = d;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
    if (Case == 11) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (a = 0; a < N; ++a) {
            c = a;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
    if (Case == 12) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          a = e;
          for (int c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
    if (Case == 13) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          c = e;
          for (int a = 0; a < N; ++a) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
    if (Case == 14) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int a = 0; a < N; ++a) {
          b = a;
          for (int c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
    if (Case == 15) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int b = 0; b < N; ++b) {
          c = b;
          for (int a = 0; a < N; ++a) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // +---+
    // | 3 |
    // +---+

    // Case 16 (43/50): (a, d). Contract (b, c, e).
    if (Case == 16) {
      a = x;
      d = y;

      for (int e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          b = e;
          c = e;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }  

    // Case 17 (46/50): (b, d). Contract (a, c, e).
    if (Case == 17) {
      b = x;
      d = y;

      for (int e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          a = e;
          c = e;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }

    // Case 18 (50/50): (d, e). Contract (a, b, c).
    if (Case == 18) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int a = 0; a < N; ++a) {
          b = a;
          c = a;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }

    value[tid] = sum;
  }
}

