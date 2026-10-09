// A line-by-line C++ translation of RAWmakase's engine 4 Basic tone code, used as the
// reference `--check` compares the GPU passes with. Sources, in
// crates/rawmakase-engine/src/develop/: pipeline/pixel.rs (tone_stage),
// pipeline/fields.rs (ExposureRamp), local_tone.rs (map, keys, blur),
// basic_tone.rs (BasicTone, photo_pivot, blocks, whites), pipeline/tone.rs (measures)
// and rawmakase-model camera_profiles (finish, profile tone curve).
#pragma once

#include "tables.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace ref {

using Rgb = std::array<float, 3>;
using Mat = std::array<Rgb, 3>; // rows

constexpr Mat TO_2020 = {{{0.627404f, 0.329283f, 0.043313f},
                          {0.069097f, 0.91954f, 0.011362f},
                          {0.016391f, 0.088013f, 0.895595f}}};
constexpr Mat FROM_2020 = {{{1.660491f, -0.587641f, -0.07285f},
                            {-0.12455f, 1.1329f, -0.008349f},
                            {-0.018151f, -0.100579f, 1.11873f}}};
constexpr Mat RGB_TO_PRO = {{{0.529345f, 0.330072f, 0.140583f},
                             {0.098374f, 0.873462f, 0.028164f},
                             {0.016883f, 0.117673f, 0.865444f}}};
constexpr Mat PRO_TO_RGB = {{{2.034075f, -0.727334f, -0.306742f},
                             {-0.228813f, 1.231731f, -0.002918f},
                             {-0.00857f, -0.153286f, 1.161856f}}};
constexpr float DNG_SHADOWS_BLACK = 0.0015f;
constexpr unsigned MAP_EDGE = 512;
constexpr float RADIUS = 0.032f;
constexpr float EPSILON = 1.5f;
constexpr float TYPICAL_PIVOT = 0.5f;
constexpr float WHITES_OFFSET = 0.24f;
constexpr float PIVOT_FIT[3] = {0.577f, 0.568f, -0.689f};
constexpr int LUT_SIZE = 1024;

inline Rgb mul(const Mat& m, const Rgb& p) {
    Rgb o;
    for (int r = 0; r < 3; ++r) o[r] = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2];
    return o;
}
inline float luminance(const Rgb& p) { return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]; }
inline float srgb_decode(float v) { return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f); }
inline float srgb_encode(float v) {
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(std::max(v, 0.f), 1.f / 2.4f) - 0.055f;
}
inline const float* T(int offset) { return tables::DATA + offset; }

// pipeline/fields.rs ExposureRamp.
struct Ramp {
    float black, slope, radius, q;
    explicit Ramp(float b) {
        black = std::clamp(b, 0.f, 0.5f);
        slope = 1.f / (1.f - black);
        radius = std::min(0.5f * black, 1.f / 16.f / slope);
        q = radius > 0.f ? slope / (4.f * radius) : 0.f;
    }
    float eval(float x) const {
        if (x <= black - radius) return 0.f;
        if (x >= black + radius) return (x - black) * slope;
        float y = x - (black - radius);
        return q * y * y;
    }
};

// camera_profiles::Profile::finish with profile_tone (no look table).
struct ProfileTone {
    std::vector<float> x, y;
    float eval(float v) const {
        float xv = std::clamp(v, 0.f, 1.f);
        size_t pp = size_t(std::upper_bound(x.begin(), x.end(), xv) - x.begin()); // partition_point(p <= x)
        size_t i = std::min(pp == 0 ? 0 : pp - 1, x.size() - 2);
        float t = std::clamp((xv - x[i]) / (x[i + 1] - x[i]), 0.f, 1.f);
        return y[i] * (1.f - t) + y[i + 1] * t;
    }
    Rgb finish(const Rgb& rgb) const {
        Rgb p = mul(RGB_TO_PRO, rgb);
        for (auto& v : p) v = std::clamp(v, 0.f, 1.f);
        float lo = std::min({p[0], p[1], p[2]});
        float hi = std::max({0.f, p[0], p[1], p[2]});
        float a = eval(lo), b = eval(hi);
        if (hi - lo > 1e-8f)
            for (auto& v : p) v = a + (b - a) * (v - lo) / (hi - lo);
        else
            p = {a, a, a};
        return mul(PRO_TO_RGB, p);
    }
};

// pipeline/pixel.rs tone_stage for engine 4 with reference curves: exposure and the
// DNG black ramp in Rec. 2020, then the profile's finish. `rgb` is the camera colour
// in linear sRGB (the POC's stand-in for camera_color).
inline Rgb tone_stage(const Rgb& rgb, float gain, float black, const ProfileTone& profile) {
    Rgb r = mul(TO_2020, rgb);
    Ramp ramp(black);
    for (auto& v : r) v = ramp.eval(v * gain);
    return profile.finish(mul(FROM_2020, r));
}

// pipeline/sampling.rs preview_source: box integration to `max` on the long edge.
inline std::vector<Rgb> reduce(const std::vector<Rgb>& px, unsigned w, unsigned h, unsigned max,
                               unsigned& ow, unsigned& oh) {
    if (std::max(w, h) <= max) {
        ow = w;
        oh = h;
        return px;
    }
    float scale = float(max) / float(std::max(w, h));
    ow = unsigned(std::lround(w * scale));
    oh = unsigned(std::lround(h * scale));
    std::vector<Rgb> out(size_t(ow) * oh);
    for (unsigned y = 0; y < oh; ++y)
        for (unsigned x = 0; x < ow; ++x) {
            unsigned x0 = x * w / ow, x1 = std::max((x + 1) * w / ow, x0 + 1);
            unsigned y0 = y * h / oh, y1 = std::max((y + 1) * h / oh, y0 + 1);
            Rgb s{0, 0, 0};
            for (unsigned yy = y0; yy < y1; ++yy)
                for (unsigned xx = x0; xx < x1; ++xx)
                    for (int c = 0; c < 3; ++c) s[c] += px[size_t(yy) * w + xx][c];
            float n = float((x1 - x0) * (y1 - y0));
            for (auto& v : s) v /= n;
            out[size_t(y) * ow + x] = s;
        }
    return out;
}

// local_tone.rs blur: mean over a (2r+1)^2 window, clamped, via f64 running sums.
inline std::vector<float> blur(const std::vector<float>& x, size_t w, size_t h, size_t r) {
    auto line = [&](size_t len, auto get) {
        auto g = [&](long i) { return double(get(size_t(std::clamp(i, 0L, long(len) - 1)))); };
        std::vector<float> out(len);
        double sum = 0;
        for (long i = -long(r); i <= long(r); ++i) sum += g(i);
        for (size_t i = 0; i < len; ++i) {
            out[i] = float(sum / double(2 * r + 1));
            sum += g(long(i) + long(r) + 1) - g(long(i) - long(r));
        }
        return out;
    };
    std::vector<float> rows(x.size()), out(x.size());
    for (size_t y = 0; y < h; ++y) {
        auto l = line(w, [&](size_t i) { return x[y * w + i]; });
        std::copy(l.begin(), l.end(), rows.begin() + long(y * w));
    }
    for (size_t c = 0; c < w; ++c) {
        auto l = line(h, [&](size_t i) { return rows[i * w + c]; });
        for (size_t y = 0; y < h; ++y) out[y * w + c] = l[y];
    }
    return out;
}

// The k-th smallest, as Rust's select_nth_unstable_by(k, total_cmp).
inline float select(std::vector<float> v, float q) {
    size_t k = size_t(float(v.size() - 1) * q);
    std::nth_element(v.begin(), v.begin() + long(k), v.end());
    return v[k];
}

struct Map {
    std::vector<float> a, b;
    float keys[2] = {0, 0};
};

// local_tone.rs LocalToneMap::from_luminance (without Clarity).
inline Map local_map(const std::vector<float>& lum, size_t w, size_t h) {
    Map m;
    std::vector<float> logs(lum.size()), sq(lum.size());
    for (size_t i = 0; i < lum.size(); ++i) {
        logs[i] = std::log2(lum[i]);
        sq[i] = logs[i] * logs[i];
    }
    size_t r = std::max<size_t>(size_t(std::lround(RADIUS * float(std::max(w, h)))), 1);
    auto mean = blur(logs, w, h, r);
    auto m2 = blur(sq, w, h, r);
    std::vector<float> a(lum.size()), b(lum.size());
    for (size_t i = 0; i < lum.size(); ++i) {
        float var = std::max(m2[i] - mean[i] * mean[i], 0.f);
        a[i] = var / (var + EPSILON);
        b[i] = mean[i] - a[i] * mean[i];
    }
    m.a = blur(a, w, h, r);
    m.b = blur(b, w, h, r);
    m.keys[0] = std::log2(select(lum, T(tables::SHADOWS)[0]));
    m.keys[1] = std::log2(select(lum, T(tables::HIGHLIGHTS)[0]));
    return m;
}

// local_tone.rs family(): a family's log2 gain at slider s and base level.
inline float family(int f, float s, float key, float base) {
    if (s == 0.f) return 0.f;
    s = std::clamp(s, -1.f, 1.f);
    const float lo = T(f)[1], hi = T(f)[2];
    const float* tables_ = T(f) + 3;
    auto bin = [&](int row) {
        const float* t = tables_ + row * 48;
        float x = std::clamp((base - key - lo) / (hi - lo) * 48.f - 0.5f, 0.f, 47.f);
        int i = std::min(int(x), 46);
        return t[i] + (t[i + 1] - t[i]) * (x - float(i));
    };
    float sv[7];
    for (int slot = 0; slot < 7; ++slot)
        sv[slot] = slot == 3 ? 0.f : T(tables::LT_SLIDER)[slot < 3 ? slot : slot - 1];
    int j = 5;
    for (int i = 0; i < 6; ++i)
        if (s <= sv[i + 1]) {
            j = i;
            break;
        }
    auto y = [&](int slot) { return slot == 3 ? 0.f : bin(slot < 3 ? slot : slot - 1); };
    float w = (s - sv[j]) / (sv[j + 1] - sv[j]);
    float y0 = y(j), y1 = y(j + 1);
    return y0 + (y1 - y0) * w;
}

// local_tone.rs Curve::new: a family's 48-bin curve at slider s (0 when s is 0).
inline std::array<float, 48> curve_table(int f, float s) {
    std::array<float, 48> table{};
    if (s == 0.f) return table;
    s = std::clamp(s, -1.f, 1.f);
    float sv[7];
    for (int slot = 0; slot < 7; ++slot)
        sv[slot] = slot == 3 ? 0.f : T(tables::LT_SLIDER)[slot < 3 ? slot : slot - 1];
    int j = 5;
    for (int i = 0; i < 6; ++i)
        if (s <= sv[i + 1]) {
            j = i;
            break;
        }
    const float* tables_ = T(f) + 3;
    auto row = [&](int slot) { return slot == 3 ? nullptr : tables_ + (slot < 3 ? slot : slot - 1) * 48; };
    const float *t0 = row(j), *t1 = row(j + 1);
    float w = (s - sv[j]) / (sv[j + 1] - sv[j]);
    for (int i = 0; i < 48; ++i) {
        float y0 = t0 ? t0[i] : 0.f, y1 = t1 ? t1[i] : 0.f;
        table[size_t(i)] = y0 + (y1 - y0) * w;
    }
    return table;
}

// basic_tone.rs.
inline std::pair<int, float> bracket(const float* xs, int n, float x) {
    x = std::clamp(x, xs[0], xs[n - 1]);
    int j = n - 2;
    for (int i = 0; i + 1 < n; ++i)
        if (x <= xs[i + 1]) {
            j = i;
            break;
        }
    return {j, std::clamp((x - xs[j]) / (xs[j + 1] - xs[j]), 0.f, 1.f)};
}
inline float interpolate(const float* xs, int n, const float* ys, float x) {
    auto [j, w] = bracket(xs, n, x);
    return ys[j] + (ys[j + 1] - ys[j]) * w;
}
using Table = std::array<std::array<float, 64>, 6>;
inline Table table(int offset) {
    Table t;
    for (int r = 0; r < 6; ++r)
        for (int i = 0; i < 64; ++i) t[r][i] = T(offset)[r * 64 + i];
    return t;
}
inline Table whites_for_highlights(float highlights) {
    float ev = interpolate(T(tables::WHITES_HIGHLIGHTS), 11, T(tables::WHITES_EXPOSURES), highlights) +
               WHITES_OFFSET;
    auto [j, w] = bracket(T(tables::WHITES_EXPOSURES), 13, ev);
    Table t = table(tables::WHITES);
    const float* wa = T(tables::WHITES_ADAPTIVE);
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < 64; ++i)
            t[3 + k][i] = wa[(j * 3 + k) * 64 + i] * (1.f - w) + wa[((j + 1) * 3 + k) * 64 + i] * w;
    return t;
}
inline float curve(const std::array<float, 64>& t, float x) {
    float f = std::clamp(x * 64.f - 0.5f, -0.5f, 63.5f);
    size_t i = size_t(std::clamp<long>(long(std::floor(f)), 0, 62));
    return std::clamp(t[i] + (t[i + 1] - t[i]) * (f - float(i)), 0.f, 1.f);
}
inline float slider(const Table& t, float s, float x) {
    if (s == 0.f) return x;
    s = std::clamp(s, -1.f, 1.f);
    float sv[7];
    for (int slot = 0; slot < 7; ++slot)
        sv[slot] = slot == 3 ? 0.f : T(tables::BT_SLIDER)[slot < 3 ? slot : slot - 1];
    int j = 5;
    for (int i = 0; i < 6; ++i)
        if (s <= sv[i + 1]) {
            j = i;
            break;
        }
    auto y = [&](int slot) { return slot == 3 ? x : curve(t[slot < 3 ? slot : slot - 1], x); };
    float w = (s - sv[j]) / (sv[j + 1] - sv[j]);
    float y0 = y(j), y1 = y(j + 1);
    return y0 + (y1 - y0) * w;
}
inline float contrast_at(float s, float pivot, float x) {
    if (s == 0.f) return x;
    auto to = [](float v) { return std::pow(srgb_decode(std::clamp(v, 0.f, 1.f)), 1.f / 2.2f); };
    auto from = [](float w) { return srgb_encode(std::pow(std::clamp(w, 0.f, 1.f), 2.2f)); };
    float k = std::log(to(T(tables::CONTRAST_PIVOT)[0])) / std::log(to(pivot));
    static const Table chart = table(tables::CONTRAST_CHART);
    float y = slider(chart, s, from(std::pow(to(x), k)));
    return from(std::pow(to(y), 1.f / k));
}
// BasicTone::new, adaptive Contrast; empty when every slider is 0.
inline std::vector<float> basic_lut(float contrast, float whites, float blacks, float pivot,
                                    const Table& whites_table) {
    if (contrast == 0.f && whites == 0.f && blacks == 0.f) return {};
    static const Table blacks_table = table(tables::BLACKS);
    std::vector<float> lut(LUT_SIZE + 1);
    float max = 0.f;
    for (int i = 0; i <= LUT_SIZE; ++i) {
        float x = float(i) / float(LUT_SIZE);
        x = slider(whites_table, whites, x);
        x = slider(blacks_table, blacks, x);
        max = std::max(max, contrast_at(contrast, pivot, x));
        lut[size_t(i)] = max;
    }
    return lut;
}
// basic_tone.rs blocks + photo_pivot, from encoded pixels.
inline float photo_pivot(const std::vector<Rgb>& encoded, size_t width, size_t height) {
    const size_t COLUMNS = 48;
    size_t rows = size_t(std::max(1.f, std::round(float(COLUMNS * height) / float(std::max<size_t>(width, 1)))));
    std::vector<float> lum;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < COLUMNS; ++c) {
            size_t x0 = c * width / COLUMNS, x1 = (c + 1) * width / COLUMNS;
            size_t y0 = r * height / rows, y1 = (r + 1) * height / rows;
            Rgb s{0, 0, 0};
            int n = 0;
            for (size_t y = y0; y < std::min(std::max(y1, y0 + 1), height); ++y)
                for (size_t x = x0; x < std::min(std::max(x1, x0 + 1), width); ++x) {
                    for (int k = 0; k < 3; ++k) s[k] += encoded[y * width + x][k];
                    ++n;
                }
            if (n == 0) continue;
            Rgb d;
            for (int k = 0; k < 3; ++k) d[k] = srgb_decode(std::clamp(s[k] / float(n), 0.f, 1.f));
            lum.push_back(srgb_encode(std::clamp(luminance(d), 1e-5f, 1.f)));
        }
    if (lum.empty()) return TYPICAL_PIVOT;
    float mean = 0;
    for (float v : lum) mean += v;
    mean /= float(lum.size());
    std::sort(lum.begin(), lum.end());
    size_t last = lum.size() - 1;
    auto pct = [&](float q) {
        float at = float(last) * q;
        size_t i = size_t(at), j = std::min(i + 1, last);
        return lum[i] + (lum[j] - lum[i]) * (at - float(i));
    };
    float middle = (pct(0.01f) + pct(0.99f)) / 2.f;
    return std::clamp(PIVOT_FIT[0] + PIVOT_FIT[1] * mean + PIVOT_FIT[2] * middle, 0.3f, 0.75f);
}

} // namespace ref
