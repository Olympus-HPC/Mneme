/****************************************************************************
 *
 * media.c, Version 1.0.0 Mon 09 Jan 2012
 *
 * ----------------------------------------------------------------------------
 *
 * CUDA EGS
 * Copyright (C) 2012 CancerCare Manitoba
 *
 * The latest version of CUDA EGS and additional information are available online at 
 * http://www.physics.umanitoba.ca/~elbakri/cuda_egs/ and http://www.lippuner.ca/cuda_egs
 *
 * CUDA EGS is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License as published by the Free Software 
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.                                       
 *                                                                           
 * CUDA EGS is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
 * details.                              
 *                                                                           
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * ----------------------------------------------------------------------------
 *
 *   Contact:
 *
 *   Jonas Lippuner
 *   Email: jonas@lippuner.ca 
 *
 ****************************************************************************/

/* This file basically implements the subroutine HATCH of the original EGSnrc code.
 * HATCH is a pretty long subroutine and calls various other subroutines which are
 * also implemented here. HATCH reads the PEGS4 file and initializes all the media
 * data. Here, this is done with the function init_media at the end of the file.
 * Most of the code was taken from the file egsnrc.mortran (v 1.72 2011/05/05) of 
 * the EGSnrc code system. Additional code was added to copy data to the device and
 * read certain files.
 */

#ifdef CUDA_EGS

#define MXGE        200
#define MXRAYFF     100
#define RAYCDFSIZE  100
#define MXELEMENT   100
#define MXSHELL     6

typedef struct element_t {
  char    symbol[2];
  double  z;
  double  wa;
  double  pz;
  double  rhoz;
} element_t;

typedef struct medium_t {
  string  *name;
  double  rho;
  double  rlc;
  double  ae, ap;
  double  ue, up;
  uint    ne;
  element_t *elements;
  int     iunrst, epstfl, iaprim;
} medium_t;

typedef struct __align__(16) rayleigh_data_t {
  float   xgrid;
  float   fcum;
  float   b_array;
  float   c_array;
} rayleigh_data_t;

double binding_energies[MXSHELL][MXELEMENT];

float2          *d_ge, *d_gmfp, *d_gbr1, *d_gbr2, *d_cohe, *d_pmax;
rayleigh_data_t *d_rayleigh_data;
uint            *d_i_array;

__constant__ float2             *ge, *gmfp, *gbr1, *gbr2, *cohe, *pmax;
__constant__ rayleigh_data_t    *rayleigh_data;
__constant__ uint               *i_array;

ulong abs_diff(ulong a, ulong b) {
  if (a > b)
    return a - b;
  else
    return b - a;
}

void read_xsec_data(const char *file, uint ndat[MXELEMENT], double2 *data[MXELEMENT]) {
  FILE *f = fopen(file, "r");
  bool ok = f != NULL;

  if (ok) {
    for (uint i = 0; i < MXELEMENT; i++) {
      uint n;
      if (fscanf(f, "%u\n", &n) != 1) {
        ok = false;
        break;
      }
      ndat[i] = n;      
      data[i] = (double2*)malloc(n * sizeof(double2));

      for (uint j = 0; j < n; j++) {
        double2 dat;
        if (fscanf(f, "%lf %lf", &dat.x, &dat.y) != 2) {
          ok = false;
          break;
        }
        data[i][j] = dat;
      }

      if (!ok)
        break;  
    }
  }

  if (f) fclose(f);

  if (!ok) printf("Could not read the data file \"%s\".\n", file);
}

void read_ff_data(const char *file, double xval[MXRAYFF], double aff[MXELEMENT][MXRAYFF]) {
  // read atomic form factors from file
  FILE *f = fopen(file, "r");

  bool ok = f != NULL;

  if (ok) {
    // read momentum transfer values
    for (uint i = 0; i < MXRAYFF; i++) {
      if (fscanf(f, "%lf", &xval[i]) != 1) {
        ok = false;
        break;
      }
    }
  }

  if (ok) {
    // read elemental form factors
    for (uint i = 0; i < MXELEMENT; i++) {
      for (uint j = 0; j < MXRAYFF; j++) {
        if (fscanf(f, "%lf", &aff[i][j]) != 1) {
          ok = false;
          break;
        }
      }
      if (!ok)
        break;
    }
  }

  if (f) fclose(f);

  if (!ok) printf("Could not read the atomic form factors file \"%s\".\n", file);
}

void heap_sort(uint n, double *values, uint *indices) {
  for (uint i = 0; i < n; i++)
    indices[i] = i + 1;

  if (n < 2)
    return;

  uint l = n / 2 + 1;
  uint idx = n;

  uint i, j;
  double last_value;
  uint last_idx;

  do {
    if (l > 1) {
      l--;
      last_value = values[l - 1];
      last_idx = l;
    }
    else {
      last_value = values[idx - 1];
      last_idx = indices[idx - 1];
      values[idx - 1] = values[0];
      indices[idx - 1] = indices[0];
      idx--;
      if (idx == 0) {
        values[0] = last_value;
        indices[0] = last_idx;
        return;
      }
    }

    i = l;
    j = 2 * l;

    do {
      if (j > idx)
        break;
      if (j < idx) {
        if (values[j - 1] < values[j])
          j++;
      }
      if (last_value < values[j - 1]) {
        values[i - 1] = values[j - 1];
        indices[i - 1] = indices[j - 1];
        i = j;
        j = 2 * j;
      }
      else
        j = idx + 1;
    } while (true);

    values[i - 1] = last_value;
    indices[i - 1] = last_idx;
  } while (true);
}

double *get_data(uint flag, uint ne, uint ndat[MXELEMENT], double2 *data[MXELEMENT], double *z_sorted, double *pz_sorted, double2 ge) {
  double *res = (double*)malloc(MXGE * sizeof(double));

  for (uint i = 0; i < MXGE; i++)
    res[i] = 0.0F;

  for (uint i = 0; i < ne; i++) {
    uint z = (uint)(z_sorted[i] + 0.5F) - 1;
    uint n = ndat[z];
    double2 *in_dat;
    double eth;

    if (flag == 0) {
      in_dat = (double2*)malloc(n * sizeof(double2));
      for (uint j = 0; j < n; j++)
        in_dat[j] = data[z][j];
    }
    else {
      in_dat = (double2*)malloc((n + 1) * sizeof(double2));
      for (uint j = 0; j < n; j++)
        in_dat[j + 1] = data[z][j];

      if (flag == 1)
        eth = 2.0 * ((double)ELECTRON_REST_MASS_FLOAT);
      else 
        eth = 4.0 * ((double)ELECTRON_REST_MASS_FLOAT);

      n++;

      for (uint j = 1; j < n; j++)
        in_dat[j].y -= 3.0 * log(1.0 - eth / exp(in_dat[j].x));

      in_dat[0] = make_double2(log(eth), in_dat[1].y);
    }

    for (uint j = 0; j < MXGE; j++) {
      double gle = ((double)j - ge.x) / ge.y;
      double e = exp(gle);
      double sig;

      if ((gle < in_dat[0].x) || (gle >= in_dat[n - 1].x)) {
        if (flag == 0) 
          printf("Energy %f is outside the available data range of %f to %f.\n", e, exp(in_dat[0].x), exp(in_dat[n - 1].x));
        else {
          if (gle < in_dat[0].x)
            sig = 0.0F;
          else
            sig = exp(in_dat[n - 1].y);
        }
      }
      else {
        uint k;
        for (k = 0; k < n - 1; k++) {
          if ((gle >= in_dat[k].x) && (gle < in_dat[k + 1].x))
            break;
        }
        double p = (gle - in_dat[k].x) / (in_dat[k + 1].x - in_dat[k].x);
        sig = exp(p * in_dat[k + 1].y + (1.0F - p) * in_dat[k].y);
      }
      if ((flag != 0) && (e > eth))
        sig *= (1.0F - eth / e) * (1.0F - eth / e) * (1.0F - eth / e);

      res[j] += pz_sorted[i] * sig;
    }

    free(in_dat);
  }

  return res;
}

double kn_sigma0(double e) {
  float con = 0.1274783851F;

  double ko = e / ((double)ELECTRON_REST_MASS_FLOAT);
  if (ko < 0.01)
    return 8.0 * con / 3.0 * (1.0 - ko * (2.0 - ko * (5.2 -13.3 * ko))) / ((double)ELECTRON_REST_MASS_FLOAT);

  double c1 = 1.0 / (ko * ko);
  double c2 = 1.0 - 2.0 * (1.0 + ko) * c1;
  double c3 = (1.0 + 2.0 * ko) * c1;
  double eps2 = 1.0;
  double eps1 = 1.0 / (1.0 + 2.0 * ko);

  return (c1 * (1.0 / eps1 - 1.0 / eps2) + c2 * log(eps2 / eps1) + eps2 * (c3 + 0.5 * eps2) - eps1 * (c3 + 0.5 * eps1)) / e * ((double)con);
}

uint read_pegs_file(const char *pegs_file, uint nmed, string *media_names, medium_t **media, bool *found) {
  FILE *pegs = fopen(pegs_file, "r");

  if (!pegs) 
    printf("Could not open the PEGS file \"%s\".\n", pegs_file);

  uint media_found = 0;

  do {
    // read line from pegs file
    char buffer[80];
    fgets(buffer, 80, pegs);
    string line(buffer);

    // here starts a medium definition
    if (line.find(" MEDIUM=") == 0) {
      string name_with_spaces = line.substr(8, 24);
      string name = "";
      // read name up to first space
      for (uint i = 0; i < 24; i++) {
        if (name_with_spaces[i] != ' ')
          name += name_with_spaces[i];
        else
          break;
      }

      // see whether this is required medium
      bool required = false;
      uint med_idx;
      for (uint i = 0; i < nmed; i++) {
        if (name == media_names[i]) {
          required = true;
          med_idx = i;
          break;
        }
      }

      if (!required)
        continue;

      // we have found the i'th required medium
      medium_t *medium = (medium_t*)malloc(sizeof(medium_t));
      medium->name = new string(name);
      medium->ne = 0;

      // read the next line containing the density, number of elements and flags
      fgets(buffer, 80, pegs);
      line = string(buffer);
      uint idx = 0;
      bool ok = true;

      do {
        size_t pos_comma = line.find(',', idx);
        if (pos_comma == string::npos)
          pos_comma = line.length();

        string entry = line.substr(idx, pos_comma - idx);
        idx = pos_comma + 1;

        size_t pos_equal = entry.find('=');
        if (pos_equal == string::npos)
          continue;

        string name_with_spaces = entry.substr(0, pos_equal);
        string name = "";
        for (uint i = 0; i < name_with_spaces.length(); i++) {
          if (name_with_spaces[i] != ' ')
            name += name_with_spaces[i];
        }
        string value = entry.substr(pos_equal + 1, entry.length() - pos_equal - 1);

        if (name == "RHO") {
          double d;
          if (sscanf(value.c_str(), "%lf", &d) != 1) {
            ok = false;
            break;
          }
          medium->rho = d;
        }
        else if (name == "NE") {
          uint u;
          if (sscanf(value.c_str(), "%u", &u) != 1) {
            ok = false;
            break;
          }
          medium->ne = u;
        }
        else if (name == "IUNRST") {
          int i;
          if (sscanf(value.c_str(), "%d", &i) != 1) {
            ok = false;
            break;
          }
          medium->iunrst = i;
        }
        else if (name == "EPSTFL") {
          int i;
          if (sscanf(value.c_str(), "%d", &i) != 1) {
            ok = false;
            break;
          }
          medium->epstfl = i;
        }
        else if (name == "IAPRIM") {
          int i;
          if (sscanf(value.c_str(), "%d", &i) != 1) {
            ok = false;
            break;
          }
          medium->iaprim = i;
        }

      } while (idx < line.length());

      if (!ok)
        continue;

      // read elements
      medium->elements = (element_t*)malloc(medium->ne * sizeof(element_t));
      for (uint i = 0; i < medium->ne; i++) {
        element_t element;

        fgets(buffer, 80, pegs);
        line = string(buffer);
        idx = 0;

        do {
          size_t pos_comma = line.find(',', idx);
          if (pos_comma == string::npos)
            pos_comma = line.length();

          string entry = line.substr(idx, pos_comma - idx);
          idx = pos_comma + 1;

          size_t pos_equal = entry.find('=');
          if (pos_equal == string::npos)
            continue;

          string name_with_spaces = entry.substr(0, pos_equal);
          string name = "";
          for (uint i = 0; i < name_with_spaces.length(); i++) {
            if (name_with_spaces[i] != ' ')
              name += name_with_spaces[i];
          }
          string value = entry.substr(pos_equal + 1, entry.length() - pos_equal - 1);

          if (name == "ASYM") {
            if (value.length() < 2) {
              ok = false;
              break;
            }
            element.symbol[0] = value[0];
            element.symbol[1] = value[1];
          }
          else if (name == "Z") {
            double d;
            if (sscanf(value.c_str(), "%lf", &d) != 1) {
              ok = false;
              break;
            }
            element.z = d;
          }
          else if (name == "A") {
            double d;
            if (sscanf(value.c_str(), "%lf", &d) != 1) {
              ok = false;
              break;
            }
            element.wa = d;
          }
          else if (name == "PZ") {
            double d;
            if (sscanf(value.c_str(), "%lf", &d) != 1) {
              ok = false;
              break;
            }
            element.pz = d;
          }
          else if (name == "RHOZ") {
            double d;
            if (sscanf(value.c_str(), "%lf", &d) != 1) {
              ok = false;
              break;
            }
            element.rhoz = d;
          }

        } while (idx < line.length());

        if (!ok)
          break;

        medium->elements[i] = element;
      }

      if (!ok)
        continue;

      // read next line that contines rlc, ae, ap, ue, up
      if (fscanf(pegs, "%lf %lf %lf %lf %lf\n", &medium->rlc, &medium->ae, &medium->ap, &medium->ue, &medium->up) != 5)
        continue;

      // save the medium and mark it found
      found[med_idx] = true;
      media[med_idx] = medium;
      media_found++;
    }

  } while ((media_found < nmed) && (!feof(pegs)));

  fclose(pegs);

  return media_found;
}

void init_user_photon(uint nmed, medium_t **media, const char *photon_xsections, const char *comp_xsections) {
  // read photon cross sections data
  uint photo_ndat[MXELEMENT];
  uint rayleigh_ndat[MXELEMENT];
  uint pair_ndat[MXELEMENT];
  uint triplet_ndat[MXELEMENT];

  double2 *photo_xsec_data[MXELEMENT];
  double2 *rayleigh_xsec_data[MXELEMENT];
  double2 *pair_xsec_data[MXELEMENT];
  double2 *triplet_xsec_data[MXELEMENT];

  string prefix = string(photon_xsections);
  read_xsec_data((prefix + "_photo.data").c_str(), photo_ndat, photo_xsec_data);
  read_xsec_data((prefix + "_rayleigh.data").c_str(), rayleigh_ndat, rayleigh_xsec_data);
  read_xsec_data((prefix + "_pair.data").c_str(), pair_ndat, pair_xsec_data);
  read_xsec_data((prefix + "_triplet.data").c_str(), triplet_ndat, triplet_xsec_data);

  for (uint i = 0; i < MXELEMENT; i++) {
    uint n = photo_ndat[i];
    uint k = 0;
    for (uint j = n - 1; j > 1; j--) {
      if (photo_xsec_data[i][j].x - photo_xsec_data[i][j - 1].x < 1.0E-5F)
        binding_energies[k++][i] = exp(photo_xsec_data[i][j].x);
      if (k >= 3)
        break;
    }
  }

  // allocate memory
  double2 *h_ge = (double2*)malloc(nmed * sizeof(double2));
  double2 *h_gmfp = (double2*)malloc(nmed * MXGE * sizeof(double2));
  double2 *h_gbr1 = (double2*)malloc(nmed * MXGE * sizeof(double2));
  double2 *h_gbr2 = (double2*)malloc(nmed * MXGE * sizeof(double2));
  double2 *h_cohe = (double2*)malloc(nmed * MXGE * sizeof(double2));

  for (uint i = 0; i < nmed; i++) {
    medium_t m = *media[i];
    h_ge[i].y = (double)(MXGE - 1) / log(m.up / m.ap);
    // indexing starts at 0 and not at 1 as in FORTRAN, i.e. subtract 1
    h_ge[i].x = -h_ge[i].y * log(m.ap);

    double sumA = 0.0F;
    double sumZ = 0.0F;

    double *z_sorted = (double*)malloc(m.ne * sizeof(double));
    for (uint j = 0; j < m.ne; j++) {
      z_sorted[j] = m.elements[j].z;
      sumA += m.elements[j].pz * m.elements[j].wa;
      sumZ += m.elements[j].pz * m.elements[j].z;
    }

    //double con1 = sumZ * m.rho / (sumA * 1.6605655F); // apparently not used
    double con2 = m.rho / (sumA * ((double)1.6605655F));

    uint *sorted = (uint*)malloc(m.ne * sizeof(uint));
    heap_sort(m.ne, z_sorted, sorted);
    // indexing starts at 0 and not at 1 as in FORTRAN, i.e. subtract 1
    for (uint j = 0; j < m.ne; j++)
      sorted[j] -= 1;

    double *pz_sorted = (double*)malloc(m.ne * sizeof(double));
    for (uint j = 0; j < m.ne; j++)
      pz_sorted[j] = m.elements[sorted[j]].pz;

    double *sig_photo = get_data(0, m.ne, photo_ndat, photo_xsec_data, z_sorted, pz_sorted, h_ge[i]);
    double *sig_rayleigh = get_data(0, m.ne, rayleigh_ndat, rayleigh_xsec_data, z_sorted, pz_sorted, h_ge[i]);
    double *sig_pair = get_data(1, m.ne, pair_ndat, pair_xsec_data, z_sorted, pz_sorted, h_ge[i]);
    double *sig_triplet = get_data(2, m.ne, triplet_ndat, triplet_xsec_data, z_sorted, pz_sorted, h_ge[i]);

    // do bound compton here

    double gle;
    double gmfp;
    double gbr1;
    double gbr2;
    double cohe;

    double gmfp_old = 0.0F;
    double gbr1_old = 0.0F;
    double gbr2_old = 0.0F;
    double cohe_old = 0.0F;

    for (uint j = 0; j < MXGE; j++) {
      gle = ((double)j - h_ge[i].x) / h_ge[i].y;
      double e = exp(gle);
      double sig_kn = sumZ * kn_sigma0(e);

      // do bound compton here

      double sig_p = sig_pair[j] + sig_triplet[j];
      double sigma = sig_kn + sig_p + sig_photo[j];
      gmfp = 1.0 / (sigma * con2);
      gbr1 = sig_p / sigma;
      gbr2 = gbr1 + sig_kn / sigma;
      cohe = sigma / (sig_rayleigh[j] + sigma);

      if (j > 0) {
        uint idx = i * MXGE + j - 1;
        h_gmfp[idx].y = (gmfp - gmfp_old) * h_ge[i].y;
        h_gmfp[idx].x = gmfp - h_gmfp[idx].y * gle;
        h_gbr1[idx].y = (gbr1 - gbr1_old) * h_ge[i].y;
        h_gbr1[idx].x = gbr1 - h_gbr1[idx].y * gle;
        h_gbr2[idx].y = (gbr2 - gbr2_old) * h_ge[i].y;
        h_gbr2[idx].x = gbr2 - h_gbr2[idx].y * gle;
        h_cohe[idx].y = (cohe - cohe_old) * h_ge[i].y;
        h_cohe[idx].x = cohe - h_cohe[idx].y * gle;
      }

      gmfp_old = gmfp;
      gbr1_old = gbr1;
      gbr2_old = gbr2;
      cohe_old = cohe;
    }

    uint idx = i * MXGE + MXGE - 1;

    h_gmfp[idx].y = h_gmfp[idx - 1].y;
    h_gmfp[idx].x = gmfp - h_gmfp[idx].y * gle;
    h_gbr1[idx].y = h_gbr1[idx - 1].y;
    h_gbr1[idx].x = gbr1 - h_gbr1[idx].y * gle;
    h_gbr2[idx].y = h_gbr2[idx - 1].y;
    h_gbr2[idx].x = gbr2 - h_gbr2[idx].y * gle;
    h_cohe[idx].y = h_cohe[idx - 1].y;
    h_cohe[idx].x = cohe - h_cohe[idx].y * gle;

    free(z_sorted);
    free(sorted);
    free(pz_sorted);

    free(sig_photo);
    free(sig_rayleigh);
    free(sig_pair);
    free(sig_triplet);
  }

  for (uint i = 0; i < MXELEMENT; i++) {
    free(photo_xsec_data[i]);
    free(rayleigh_xsec_data[i]);
    free(pair_xsec_data[i]);
    free(triplet_xsec_data[i]);
  }

  // convert to floats
  float2 *h_ge_f = (float2*)malloc(nmed * sizeof(float2));
  float2 *h_gmfp_f = (float2*)malloc(nmed * MXGE * sizeof(float2));
  float2 *h_gbr1_f = (float2*)malloc(nmed * MXGE * sizeof(float2));
  float2 *h_gbr2_f = (float2*)malloc(nmed * MXGE * sizeof(float2));
  float2 *h_cohe_f = (float2*)malloc(nmed * MXGE * sizeof(float2));

  for (uint i = 0; i < nmed; i++)
    h_ge_f[i] = make_float2((float)h_ge[i].x, (float)h_ge[i].y);

  for (uint i = 0; i < nmed * MXGE; i++) {
    h_gmfp_f[i] = make_float2((float)h_gmfp[i].x, (float)h_gmfp[i].y);
    h_gbr1_f[i] = make_float2((float)h_gbr1[i].x, (float)h_gbr1[i].y);
    h_gbr2_f[i] = make_float2((float)h_gbr2[i].x, (float)h_gbr2[i].y);
    h_cohe_f[i] = make_float2((float)h_cohe[i].x, (float)h_cohe[i].y);
  }

  // free host double memory
  free(h_ge);
  free(h_gmfp);
  free(h_gbr1);
  free(h_gbr2);
  free(h_cohe);

  // allocate device memory
  cudaMalloc(&d_ge, nmed * sizeof(float2));
  cudaMalloc(&d_gmfp, nmed * MXGE * sizeof(float2));
  cudaMalloc(&d_gbr1, nmed * MXGE * sizeof(float2));
  cudaMalloc(&d_gbr2, nmed * MXGE * sizeof(float2));
  cudaMalloc(&d_cohe, nmed * MXGE * sizeof(float2));

  // copy data to device
  cudaMemcpy(d_ge, h_ge_f, nmed * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gmfp, h_gmfp_f, nmed * MXGE * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gbr1, h_gbr1_f, nmed * MXGE * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gbr2, h_gbr2_f, nmed * MXGE * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cohe, h_cohe_f, nmed * MXGE * sizeof(float2), cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(ge, &d_ge, sizeof(float2*));
  cudaMemcpyToSymbol(gmfp, &d_gmfp, sizeof(float2*));
  cudaMemcpyToSymbol(gbr1, &d_gbr1, sizeof(float2*));
  cudaMemcpyToSymbol(gbr2, &d_gbr2, sizeof(float2*));
  cudaMemcpyToSymbol(cohe, &d_cohe, sizeof(float2*));

  // free host single memory
  free(h_ge_f);
  free(h_gmfp_f);
  free(h_gbr1_f);
  free(h_gbr2_f);
  free(h_cohe_f);
}

void init_rayleigh_data(uint nmed, medium_t **media) {
  double xval[MXRAYFF];
  double aff[MXELEMENT][MXRAYFF];

  read_ff_data(atomic_ff_file, xval, aff);

  // allocate memory
  double *h_ff = (double*)malloc(MXRAYFF * nmed * sizeof(double));
  double *h_xgrid = (double*)malloc(MXRAYFF * nmed * sizeof(double));
  double *h_fcum = (double*)malloc(MXRAYFF * nmed * sizeof(double)); 
  double *h_b_array = (double*)malloc(MXRAYFF * nmed * sizeof(double)); 
  double *h_c_array = (double*)malloc(MXRAYFF * nmed * sizeof(double)); 
  uint *h_i_array = (uint*)malloc(RAYCDFSIZE * nmed * sizeof(uint)); 
  double *h_pe_array = (double*)malloc(MXGE * nmed * sizeof(double)); 
  double2 *h_pmax = (double2*)malloc(MXGE * nmed * sizeof(double2));

  for (uint i = 0; i < nmed; i++) {
    // calculate form factor using independent atom model
    for (uint j = 0; j < MXRAYFF; j++) {
      double ff_val = 0.0;
      h_xgrid[i * MXRAYFF + j] = xval[j];

      for (uint k = 0; k < media[i]->ne; k ++) {
        uint z = (uint)media[i]->elements[k].z - 1;
        ff_val += media[i]->elements[k].pz * aff[z][j] * aff[z][j];
      }

      h_ff[i * MXRAYFF + j] = sqrt(ff_val);
    }

    // to avoid log(0)
    /*if (*((ulong*)&h_xgrid[i * MXRAYFF]) == 0) {
      ulong zero = 1;
      h_xgrid[i * MXRAYFF] = *((double*)&zero);
      }*/
    if (h_xgrid[i * MXRAYFF] < 1E-6)
      h_xgrid[i * MXRAYFF] = ((double)0.0001F);

    // calculate rayleigh data (subroutine prepare_rayleigh_data)

    double2 ge;
    ge.y = (double)(MXGE - 1) / log(media[i]->up / media[i]->ap);
    // indexing starts at 0 and not at 1 as in FORTRAN, i.e. subtract 1
    ge.x = -ge.y * log(media[i]->ap);

    double emin = exp(-ge.x / ge.y);
    double emax = exp(((double)MXGE - 1.0 - ge.x) / ge.y);

    // to avoid log (0)
    for (uint j = 0; j < MXRAYFF; j++) {
      if (*((ulong*)&h_ff[i * MXRAYFF + j]) == 0) {
        ulong zero = 1;
        h_ff[i * MXRAYFF + j] = *((double*)&zero);
      }
    }

    /**********************************************************
     * Calculate the cumulative distribution
     *********************************************************/
    double sum0 = 0.0;
    h_fcum[i * MXRAYFF] = 0.0;

    for (uint j = 0; j < MXRAYFF - 1; j++) {
      double b = log(h_ff[i * MXRAYFF + j + 1] / h_ff[i * MXRAYFF + j]) / log(h_xgrid[i * MXRAYFF + j + 1] / h_xgrid[i * MXRAYFF + j]);
      h_b_array[i * MXRAYFF + j] = b;
      double x1 = h_xgrid[i * MXRAYFF + j];
      double x2 = h_xgrid[i * MXRAYFF + j + 1];
      double pow_x1 = pow(x1, 2.0 * b);
      double pow_x2 = pow(x2, 2.0 * b);
      sum0 += h_ff[i * MXRAYFF + j] * h_ff[i * MXRAYFF + j] * (x2 * x2 * pow_x2 - x1 * x1 * pow_x1) / ((1.0 + b) * pow_x1);
      h_fcum[i * MXRAYFF + j + 1] = sum0;
    }

    h_b_array[i * MXRAYFF + MXRAYFF - 1] = 0.0;

    /*************************************************************
     * Now the maximum cumulative probability as a function of
     * incident photon energy. We have xmax = 2*E*20.60744/m, so
     * pe_array(E) = fcum(xmax)
     **************************************************************/
    //double dle = log(media[i]->up / media[i]->ap) / ((double)MXGE - 1.0);
    double dle = log(emax / emin) / ((double)MXGE - 1.0);
    uint idx = 1;

    for (uint j = 1; j <= MXGE; j++) {
      //double e = media[i]->ap * exp(dle * ((double)j - 1.0));
      double e = emin * exp(dle * ((double)j - 1.0));
      double xmax = 20.607544 * 2.0 * e / ELECTRON_REST_MASS_DOUBLE;
      uint k = 1;
      for (k = 1; k <= MXRAYFF - 1; k++) {
        if ((xmax >= h_xgrid[i * MXRAYFF + k - 1]) && (xmax < h_xgrid[i * MXRAYFF + k]))
          break;
      }
      idx = k;
      double b = h_b_array[i * MXRAYFF + idx - 1];
      double x1 = h_xgrid[i * MXRAYFF + idx - 1];
      double x2 = xmax;
      double pow_x1 = pow(x1, 2.0 * b);
      double pow_x2 = pow(x2, 2.0 * b);
      h_pe_array[i * MXGE + j - 1] = h_fcum[i * MXRAYFF + idx - 1] + h_ff[i * MXRAYFF + idx - 1] * h_ff[i * MXRAYFF + idx - 1] * 
        (x2 * x2 * pow_x2 - x1 * x1 * pow_x1) / ((1.0 + b) * pow_x1);
    }

    h_i_array[i * RAYCDFSIZE + RAYCDFSIZE - 1] = idx;

    /***********************************************************************
     * Now renormalize data so that pe_array(emax)=1
     * Note that we make pe_array(j) slightly larger so that fcum(xmax) is
     * never underestimated when interpolating
     ***********************************************************************/
    double anorm = 1.0 / sqrt(h_pe_array[i * MXGE + MXGE - 1]);
    double anorm1 = 1.005 / h_pe_array[i * MXGE + MXGE - 1];
    double anorm2 = 1.0 / h_pe_array[i * MXGE + MXGE - 1];

    for (uint j = 0; j < MXGE; j++) {
      h_pe_array[i * MXGE + j] *= anorm1;
      if (h_pe_array[i * MXGE + j] > 1.0)
        h_pe_array[i * MXGE + j] = 1.0;
    }

    for (uint j = 0; j < MXRAYFF; j++) {
      h_ff[i * MXRAYFF + j] *= anorm;
      h_fcum[i * MXRAYFF + j] *= anorm2;
      h_c_array[i * MXRAYFF + j] = (1.0 + h_b_array[i * MXRAYFF + j]) / 
        ((h_xgrid[i * MXRAYFF + j] * h_ff[i * MXRAYFF + j]) * (h_xgrid[i * MXRAYFF + j] * h_ff[i * MXRAYFF + j]));
    }

    /***********************************************************************
     * Now prepare uniform cumulative bins
     ***********************************************************************/
    double dw = 1.0 / ((double)RAYCDFSIZE - 1.0);
    double xold = h_xgrid[i * MXRAYFF + 0];
    uint ibin = 1;
    double b = h_b_array[i * MXRAYFF + 0];
    double pow_x1 = pow(h_xgrid[i * MXRAYFF + 0], 2.0 * b);
    h_i_array[i * MXRAYFF + 0] = 1;

    for (uint j = 2; j <= RAYCDFSIZE - 1; j++) {
      double w = dw;
      do {
        double x1 = xold;
        double x2 = h_xgrid[i * MXRAYFF + ibin];
        double t = x1 * x1 * pow(x1, 2.0 * b);
        double pow_x2 = pow(x2, 2.0 * b);
        double aux = h_ff[i * MXRAYFF + ibin - 1] * h_ff[i * MXRAYFF + ibin - 1] * (x2 * x2 * pow_x2 - t) / ((1.0 + b) * pow_x1);
        if (aux > w) {
          xold = exp(log(t + w * (1.0 + b) * pow_x1 / (h_ff[i * MXRAYFF + ibin - 1] * h_ff[i * MXRAYFF + ibin - 1])) / (2.0 + 2.0 * b));
          h_i_array[i * RAYCDFSIZE + j - 1] = ibin;
          break;
        }
        w -= aux;
        xold = x2;
        ibin++;
        b = h_b_array[i * MXRAYFF + ibin - 1];
        pow_x1 = pow(xold, 2.0 * b);
      } while (true);
    }

    /*************************************************************************
     * Change definition of b_array because that's what is needed at run time
     **************************************************************************/
    for (uint j = 0; j < MXRAYFF; j++)
      h_b_array[i * MXRAYFF + j] = 0.5 / (1.0 + h_b_array[i * MXRAYFF + j]);

    // prepare coefficients for pmax interpolation
    //dle = log(media[i]->up / media[i]->ap) / ((double)MXGE - 1.0);
    //double dlei = 1.0 / dle;

    for (uint j = 0; j < MXGE - 1; j++) {
      double gle = ((double)j - ge.x) / ge.y;
      h_pmax[i * MXGE + j].y = (h_pe_array[i * MXGE + j + 1] - h_pe_array[i * MXGE + j]) * ge.y;
      h_pmax[i * MXGE + j].x = h_pe_array[i * MXGE + j] - h_pmax[i * MXGE + j].y * gle;
    }

    h_pmax[i * MXGE + MXGE - 1] = h_pmax[i * MXGE + MXGE - 2];

  }

  // convert to floats
  rayleigh_data_t *h_rayleigh_data = (rayleigh_data_t*)malloc(nmed * MXRAYFF * sizeof(rayleigh_data_t));
  float2 *h_pmax_f = (float2*)malloc(nmed * MXGE * sizeof(float2));

  for (uint i = 0; i < nmed * MXRAYFF; i++) {
    h_rayleigh_data[i].xgrid = (float)h_xgrid[i];
    h_rayleigh_data[i].fcum = (float)h_fcum[i];
    h_rayleigh_data[i].b_array = (float)h_b_array[i];
    h_rayleigh_data[i].c_array = (float)h_c_array[i];
  }

  for (uint i = 0; i < nmed * MXGE; i++)
    h_pmax_f[i] = make_float2((float)h_pmax[i].x, (float)h_pmax[i].y);

  // free host double memory
  free(h_ff);
  free(h_xgrid);
  free(h_fcum);
  free(h_b_array);
  free(h_c_array);
  free(h_pe_array);
  free(h_pmax);

  // allocate device memory
  cudaMalloc(&d_rayleigh_data, nmed * MXRAYFF * sizeof(rayleigh_data_t));
  cudaMalloc(&d_i_array, nmed * RAYCDFSIZE * sizeof(uint));
  cudaMalloc(&d_pmax, nmed * MXGE * sizeof(float2));

  // copy data to device
  cudaMemcpy(d_rayleigh_data, h_rayleigh_data, nmed * MXRAYFF * sizeof(rayleigh_data_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_i_array, h_i_array, nmed * RAYCDFSIZE * sizeof(uint), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pmax, h_pmax_f, nmed * MXGE * sizeof(float2), cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(rayleigh_data, &d_rayleigh_data, sizeof(rayleigh_data_t*));
  cudaMemcpyToSymbol(i_array, &d_i_array, sizeof(int*));
  cudaMemcpyToSymbol(pmax, &d_pmax, sizeof(float2*));

  // free host single memory
  free(h_rayleigh_data);
  free(h_i_array);
  free(h_pmax_f);
}

medium_t** init_media(uint nmed, string *media_names) {
  // read the data of the required media from the pegs file    
  medium_t **media = (medium_t**)malloc(nmed * sizeof(medium_t*));

  bool *found = (bool*)malloc(nmed * sizeof(bool));
  for (uint i = 0; i < nmed; i++)
    found[i] = false;

  uint media_found = read_pegs_file(pegs_file, nmed, media_names, media, found);

  // did not find all media
  if (media_found < nmed) {
    if (nmed - media_found > 1)
      printf("\nERROR: The following media were not found or could not be read from the PEGS file:");
    else
      printf("\nERROR: The following mediun was not found or could not be read from the PEGS file:");

    for (uint i = 0; i < nmed; i++) {
      if (!found[i])
        printf(" %s", media_names[i].c_str());
    }

    free(found);

    exit(10029);
  }

  free(found);

  // at this point we have found and read all required media

  // init the photon data using the specified cross sections files
  init_user_photon(nmed, media, photon_xsections, "");

  // init rayleigh data
  init_rayleigh_data(nmed, media);

  return media;
}

#endif
