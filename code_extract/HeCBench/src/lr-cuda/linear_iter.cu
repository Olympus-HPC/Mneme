#include <math.h>
#include "linear.h"

static void r_squared(
  linear_param_t * params,
  data_t * dataset, 
  sum_t * sumset, 
  result_t * response) 
{
  float mean = sumset->y / params->size;
  rsquared_t dist = {0.f, 0.f};
  float y_estimated = 0;

  for (size_t i = 0; i < params->size; i++) {
    dist.x += powf((dataset[i].y - mean), 2.f);
    y_estimated = dataset[i].x * response->a1 + response->a0;
    dist.y += powf((y_estimated - mean), 2.f);
  }

  // LOG_RSQUARED_T(dist);
  response->rsquared = dist.y / dist.x * 100;
}

void iterative_regression(
  linear_param_t * params, 
  data_t * dataset, 
  result_t * response) 
{
  sum_t sumset = {0, 0, 0, 0};

  for (size_t i = 0; i < params->size; i++) {
    sumset.x += dataset[i].x;
    sumset.y += dataset[i].y;
    sumset.z += dataset[i].x * dataset[i].y;
    sumset.w += powf(dataset[i].x, 2.f);
  }
  
  double det = params->size * sumset.w - powf(sumset.x, 2.f);

  response->a0 = (sumset.y * sumset.w - sumset.x * sumset.z) / det;
  response->a1 = (params->size * sumset.z - sumset.x * sumset.y) / det;

  r_squared(params, dataset, &sumset, response);
}
