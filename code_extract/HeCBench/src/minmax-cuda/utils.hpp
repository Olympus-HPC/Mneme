
template <typename T>
auto generate_rects(const T size)
{
  auto const phi = static_cast<T>((1 + std::sqrt(5)) * .5);

  vec_2d<T> tl{T{0}, T{0}}; // top left
  vec_2d<T> br{size, size}; // bottom right
  vec_2d<T> area = br - tl;

  std::size_t num_points = 0;
  std::vector<std::tuple<std::size_t, vec_2d<T>, vec_2d<T>>> rects{};

  do {
    switch (rects.size() % 4) {
      case 0: br.x = tl.x - (tl.x - br.x) / phi; break;
      case 1: br.y = tl.y - (tl.y - br.y) / phi; break;
      case 2: tl.x = tl.x + (br.x - tl.x) / phi; break;
      case 3: tl.y = tl.y + (br.y - tl.y) / phi; break;
    }

    area = br - tl;

    auto num_points_in_rect = static_cast<std::size_t>(std::sqrt(area.x * area.y * 1'000'000));

    rects.push_back(std::make_tuple(num_points_in_rect, tl, br));

    num_points += num_points_in_rect;
  } while (area.x > 1 && area.y > 1);

  return std::make_pair(num_points, std::move(rects));
}

/**
 * @brief Generate a random point within a window of [minXY, maxXY]
 */
template <typename T>
vec_2d<T> random_point(vec_2d<T> minXY, vec_2d<T> maxXY)
{
  auto x = minXY.x + (maxXY.x - minXY.x) * rand() / static_cast<T>(RAND_MAX);
  auto y = minXY.y + (maxXY.y - minXY.y) * rand() / static_cast<T>(RAND_MAX);
  return vec_2d<T>{x, y};
}

template <typename T>
std::pair<std::size_t, std::vector<vec_2d<T>>> generate_points(const T size)
{
  auto const [total_points, rects] = generate_rects(size);

  srand(123);
  std::size_t point_offset{0};
  std::vector<vec_2d<T>> h_points(total_points);
  for (auto const& rect : rects) {
    auto const num_points_in_rect = std::get<0>(rect);
    auto const tl                 = std::get<1>(rect);
    auto const br                 = std::get<2>(rect);
    auto points_begin             = h_points.begin() + point_offset;
    auto points_end               = points_begin + num_points_in_rect;
    std::generate(points_begin, points_end, [&]() { return random_point<T>(tl, br); });
    point_offset += num_points_in_rect;
  }
  return {total_points, h_points};
}
