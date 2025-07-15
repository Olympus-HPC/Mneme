#include <algorithm>

void reference (
    const float *__restrict img,
    const float *__restrict kernels,
    const float *__restrict offsets_h,
    const float *__restrict offsets_v,
          float *__restrict output,
    const params p,
    const int offset_unit,
    const int padding)
{
  const int dim_b = p.output_dim_b;
  const int dim_c = p.output_dim_c;
  const int dim_h = p.output_dim_h;
  const int dim_w = p.output_dim_w;
  const int kernels_size = p.kernel_size;
  const int img_w = p.image_w;
  const int img_h = p.image_h;

  const int k_size = (int)sqrtf(float(kernels_size));
  const int w = img_w - 2 * padding;
  const int h = img_h - 2 * padding;

  for (int idb = 0; idb < dim_b; idb++)
  for (int idc = 0; idc < dim_c; idc++)
  for (int idy = 0; idy < dim_h; idy++)
  for (int idx = 0; idx < dim_w; idx++) {

    float result = 0;
    for(int k_y = 0; k_y < k_size; ++k_y)
    {
      for(int k_x = 0; k_x < k_size; ++k_x)
      {
        const float offset_h = offsets_h(idb,k_size * k_y + k_x,idy,idx) * offset_unit;
        const float offset_v = offsets_v(idb,k_size * k_y + k_x,idy,idx) * offset_unit;

        const float p_x = static_cast<float>(idx + 0.5f) / dim_w * w + k_x + offset_h - 0.5f;
        const float p_y = static_cast<float>(idy + 0.5f) / dim_h * h + k_y + offset_v - 0.5f;
        const float alpha = p_x - floorf(p_x);
        const float beta = p_y - floorf(p_y);

        const int xL = std::max(std::min(int(floorf(p_x)), w + 2 * padding - 1), 0);
        const int xR = std::max(std::min(xL + 1, w + 2 * padding - 1), 0);
        const int yT = std::max(std::min(int(floorf(p_y)), h + 2 * padding - 1), 0);
        const int yB = std::max(std::min(yT + 1, h + 2 * padding - 1), 0);

        float val = (1.f - alpha) * (1.f - beta) * img(idb,idc,yT,xL);
        val += alpha * (1.f - beta) * img(idb,idc,yT,xR);
        val += (1.f - alpha) * beta * img(idb,idc,yB,xL);
        val += alpha * beta * img(idb,idc,yB,xR);
        result += val * kernels(idb,k_size * k_y + k_x,idy,idx);
      }
    }
    output(idb,idc,idy,idx) = result;
  }
}

