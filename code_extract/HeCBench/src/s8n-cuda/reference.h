void cube_select(int b, int n, int radius, const int* in, int* out) {
  for (int batch_idx = 0; batch_idx < b; batch_idx++) {
    auto xyz = in + batch_idx * n * 3;
    auto idx_out = out + batch_idx * n * 8;
    int temp_dist[8];
    for(int i = 0; i < n; i++) {
      int x = xyz[i * 3];
      int y = xyz[i * 3 + 1];
      int z = xyz[i * 3 + 2];
      for(int j = 0; j < 8;j ++) {
        temp_dist[j] = radius;
        idx_out[i * 8 + j] = i; // if not found, just return itself..
      }
      for(int j = 0; j < n; j ++) {
        if(i == j) continue;
        int tx = xyz[j * 3];
        int ty = xyz[j * 3 + 1];
        int tz = xyz[j * 3 + 2];
        int dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
        if(dist > radius) continue;
        int _x = (tx > x);
        int _y = (ty > y);
        int _z = (tz > z);
        int temp_idx = _x * 4 + _y * 2 + _z;
        if(dist < temp_dist[temp_idx]) {
          idx_out[i * 8 + temp_idx] = j;
          temp_dist[temp_idx] = dist;
        }
      }
    }
  }
}

void cube_select_two(int b, int n, int radius, const int* in, int* out) {
  for (int batch_idx = 0; batch_idx < b; batch_idx++) {
    auto xyz = in + batch_idx * n * 3;
    auto idx_out = out + batch_idx * n * 16;
    int temp_dist[16];
    for(int i = 0; i < n; i++) {
      int x = xyz[i * 3];
      int y = xyz[i * 3 + 1];
      int z = xyz[i * 3 + 2];
      for(int j = 0; j < 16;j ++) {
        temp_dist[j] = radius;
        idx_out[i * 16 + j] = i; // if not found, just return itself..
      }
      for(int j = 0; j < n; j ++) {
        if(i == j) continue;
        int tx = xyz[j * 3];
        int ty = xyz[j * 3 + 1];
        int tz = xyz[j * 3 + 2];
        int dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
        if(dist > radius) continue;
        int _x = (tx > x);
        int _y = (ty > y);
        int _z = (tz > z);
        int temp_idx = _x * 8 + _y * 4 + _z * 2;
        bool flag = false;
        for(int k = 0; k < 2; k ++) {
          if (dist < temp_dist[temp_idx + k]) {
            flag = true;
          }
          if (flag) {
            for (int kk = 1; kk >= k + 1; kk --) {
              idx_out[i * 16 + temp_idx + kk] = idx_out[i * 16 + temp_idx + kk - 1];
              temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
            }
            idx_out[i * 16 + temp_idx + k] = j;
            temp_dist[temp_idx + k] = dist;
            break;
          }
        }
      }
    }
  }
}

void cube_select_four(int b, int n, int radius, const int* in, int* out) {
  for (int batch_idx = 0; batch_idx < b; batch_idx++) {
    auto xyz = in + batch_idx * n * 3;
    auto idx_out = out + batch_idx * n * 32;
    int temp_dist[32];
    for(int i = 0; i < n; i++) {
      int x = xyz[i * 3];
      int y = xyz[i * 3 + 1];
      int z = xyz[i * 3 + 2];
      for(int j = 0; j < 32;j ++) {
        temp_dist[j] = radius;
        idx_out[i * 32 + j] = i; // if not found, just return itself..
      }
      for(int j = 0; j < n; j ++) {
        if(i == j) continue;
        int tx = xyz[j * 3];
        int ty = xyz[j * 3 + 1];
        int tz = xyz[j * 3 + 2];
        int dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
        if(dist > radius) continue;
        int _x = (tx > x);
        int _y = (ty > y);
        int _z = (tz > z);
        int temp_idx = _x * 16 + _y * 8 + _z * 4;
        bool flag = false;
        for(int k = 0; k < 4; k ++) {
          if (dist < temp_dist[temp_idx + k]) {
            flag = true;
          }
          if (flag) {
            for (int kk = 3; kk >= k + 1; kk --) {
              idx_out[i * 32 + temp_idx + kk] = idx_out[i * 32 + temp_idx + kk - 1];
              temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
            }
            idx_out[i * 32 + temp_idx + k] = j;
            temp_dist[temp_idx + k] = dist;
            break;
          }
        }
      }
    }
  }
}
