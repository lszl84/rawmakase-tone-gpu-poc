// Shared by every develop pass: bindings, parameters, and the GLSL ports of
// RAWmakase's engine 4 tone code (see src/tone_ref.h for the C++ reference and the
// Rust sources).

#include "tables.glsl"

// The fragment stage only reads the work buffers.
#ifdef FRAGMENT_STAGE
#define WRITABLE readonly
#else
#define WRITABLE
#endif

layout(std430, set = 0, binding = 0) readonly buffer Tables { float T[]; };
// The profile tone curve: x values, then y values (`tone_count` each).
layout(std430, set = 0, binding = 1) readonly buffer Tone { float tone[]; };
// The photo reduced to 512 pixels on the long edge, linear sRGB.
layout(std430, set = 0, binding = 2) readonly buffer Small { vec4 px[]; };
// Work arrays, `n` floats each (see the S_ offsets).
layout(std430, set = 0, binding = 3) WRITABLE buffer Scratch { float S[]; };
layout(std430, set = 0, binding = 4) WRITABLE buffer Measures {
    float pivot;      // the photo's Contrast pivot
    float highlights; // 98th percentile of encoded luminance, for Whites
    float key_shadows;
    float key_highlights;
    uint prefix[4];   // radix select state, per selection
    uint kth[4];
    uint hist[1024];
};
// The Basic curve, then Shadows' and Highlights' curves at their slider values
// (local_tone.rs Curve::new, 48 bins each, built on the CPU per change).
layout(std430, set = 0, binding = 5) WRITABLE buffer Lut {
    float lut[1025];
    float shadows_curve[48];
    float highlights_curve[48];
};

layout(push_constant) uniform Params {
    vec4 rect;       // the photo's rectangle in NDC (draw only)
    float gain;      // 2^(Exposure + camera exposure + profile exposure)
    float black;     // the DNG ramp's black point at that exposure
    float gain0;     // the same without the user's Exposure, for the Contrast pivot
    float black0;
    float contrast;  // sliders, -1..1
    float highlights_s;
    float shadows_s;
    float whites_s;
    float blacks_s;
    uint n;          // small image pixels, width, height
    uint w;
    uint h;
    uint flags;
    int shift;       // radix select round
    int radius;      // box radius of the guided filter
    uint tone_count;
    uint block_rows; // rows of 48-column blocks for the pivot
} pc;

const uint F_MAP = 1u;            // Shadows or Highlights set: apply the local map
const uint F_BASIC = 2u;          // Contrast, Whites or Blacks set: apply the curve
const uint F_ADAPTIVE_WHITES = 4u;
const uint F_SECOND = 8u;         // the guided filter's second box pass
const uint F_TONE_UNIFORM = 16u;  // the tone curve's knots are i / (count - 1): index directly

// Offsets of the work arrays in S.
uint S_L() { return 0u; }            // luminance of the toned small image
uint S_E() { return pc.n; }          // its encoded luminance
uint S_LOG() { return 2u * pc.n; }   // log2 luminance
uint S_R1() { return 3u * pc.n; }    // row means
uint S_R2() { return 4u * pc.n; }
uint S_A() { return 5u * pc.n; }     // guided filter a, b before their blur
uint S_B() { return 6u * pc.n; }
uint S_FA() { return 7u * pc.n; }    // the map: blurred a, b
uint S_FB() { return 8u * pc.n; }
uint S_BL() { return 9u * pc.n; }    // block luminance for the pivot (4096)

// Row-major matrices: v * mat3(rows) multiplies by the matrix.
const mat3 TO_2020 = mat3(vec3(0.627404, 0.329283, 0.043313), vec3(0.069097, 0.91954, 0.011362),
                          vec3(0.016391, 0.088013, 0.895595));
const mat3 FROM_2020 = mat3(vec3(1.660491, -0.587641, -0.07285), vec3(-0.12455, 1.1329, -0.008349),
                            vec3(-0.018151, -0.100579, 1.11873));
const mat3 RGB_TO_PRO = mat3(vec3(0.529345, 0.330072, 0.140583), vec3(0.098374, 0.873462, 0.028164),
                             vec3(0.016883, 0.117673, 0.865444));
const mat3 PRO_TO_RGB = mat3(vec3(2.034075, -0.727334, -0.306742), vec3(-0.228813, 1.231731, -0.002918),
                             vec3(-0.00857, -0.153286, 1.161856));
vec3 mulr(mat3 m, vec3 p) { return p * m; }

float srgb_decode(float v) { return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4); }
float srgb_encode(float v) { return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(max(v, 0.0), 1.0 / 2.4) - 0.055; }
float luminance(vec3 p) { return 0.2126 * p.r + 0.7152 * p.g + 0.0722 * p.b; }

// pipeline/fields.rs ExposureRamp.
float ramp(float x, float b) {
    float black = clamp(b, 0.0, 0.5);
    float slope = 1.0 / (1.0 - black);
    float radius = min(0.5 * black, 1.0 / 16.0 / slope);
    float q = radius > 0.0 ? slope / (4.0 * radius) : 0.0;
    if (x <= black - radius) return 0.0;
    if (x >= black + radius) return (x - black) * slope;
    float y = x - (black - radius);
    return q * y * y;
}

// The profile tone curve: piecewise linear, found by binary search as
// partition_point does.
float profile_tone(float v) {
    float x = clamp(v, 0.0, 1.0);
    int count = int(pc.tone_count);
    int i;
    if ((pc.flags & F_TONE_UNIFORM) != 0u) {
        // The last knot at or below x, as the search finds it: exact for knots i/2^k.
        i = min(int(x * float(count - 1)), count - 2);
    } else {
        int lo = 0, hi = count;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (tone[mid] <= x) lo = mid + 1;
            else hi = mid;
        }
        i = min(max(lo - 1, 0), count - 2);
    }
    float t = clamp((x - tone[i]) / (tone[i + 1] - tone[i]), 0.0, 1.0);
    return tone[count + i] * (1.0 - t) + tone[count + i + 1] * t;
}

// DNG RGBTone: the curve maps the largest and smallest channel, the middle one keeps
// its relative position.
vec3 rgb_tone(vec3 p, float a, float b) {
    float lo = min(min(p.r, p.g), p.b);
    float hi = max(max(max(p.r, p.g), p.b), 0.0);
    return hi - lo > 1e-8 ? a + (b - a) * (p - lo) / (hi - lo) : vec3(a);
}

// pipeline/pixel.rs tone_stage, engine 4 with reference curves, for linear sRGB input.
vec3 tone_stage(vec3 rgb, float gain, float black) {
    vec3 r = mulr(TO_2020, rgb) * gain;
    r = vec3(ramp(r.r, black), ramp(r.g, black), ramp(r.b, black));
    vec3 p = clamp(mulr(RGB_TO_PRO, mulr(FROM_2020, r)), 0.0, 1.0);
    float lo = min(min(p.r, p.g), p.b);
    float hi = max(max(max(p.r, p.g), p.b), 0.0);
    p = rgb_tone(p, profile_tone(lo), profile_tone(hi));
    return mulr(PRO_TO_RGB, p);
}

// Slider positions with the identity inserted at slot 3, as the Rust bracket builds them.
float slot_value(int values, int slot) {
    return slot == 3 ? 0.0 : T[values + (slot < 3 ? slot : slot - 1)];
}
int bracket_slot(int values, float s) {
    for (int i = 0; i < 6; ++i)
        if (s <= slot_value(values, i + 1)) return i;
    return 5;
}

// local_tone.rs family(): log2 gain of Shadows or Highlights at a base level.
float family(int f, float s, float key, float base) {
    if (s == 0.0) return 0.0;
    s = clamp(s, -1.0, 1.0);
    float lo = T[f + 1], hi = T[f + 2];
    float x = clamp((base - key - lo) / (hi - lo) * 48.0 - 0.5, 0.0, 47.0);
    int i = min(int(x), 46);
    float fx = x - float(i);
    int j = bracket_slot(T_LT_SLIDER, s);
    float y[2];
    for (int k = 0; k < 2; ++k) {
        int slot = j + k;
        if (slot == 3) {
            y[k] = 0.0;
        } else {
            int t = f + 3 + (slot < 3 ? slot : slot - 1) * 48;
            y[k] = T[t + i] + (T[t + i + 1] - T[t + i]) * fx;
        }
    }
    float w = (s - slot_value(T_LT_SLIDER, j)) / (slot_value(T_LT_SLIDER, j + 1) - slot_value(T_LT_SLIDER, j));
    return y[0] + (y[1] - y[0]) * w;
}

// local_tone.rs Curve::eval: a family's built curve at a base level.
float curve_eval(int f, float key, float base, bool shadows) {
    float lo = T[f + 1], hi = T[f + 2];
    float x = clamp((base - key - lo) / (hi - lo) * 48.0 - 0.5, 0.0, 47.0);
    int i = min(int(x), 46);
    float a = shadows ? shadows_curve[i] : highlights_curve[i];
    float b = shadows ? shadows_curve[i + 1] : highlights_curve[i + 1];
    return a + (b - a) * (x - float(i));
}

// basic_tone.rs: the Basic curve's table value at bin `i`, of row `row` of `table`;
// Whites' positive rows follow the photo's highlights when adaptive.
float basic_value(int table, int row, int i) {
    if (table == T_WHITES && row >= 3 && (pc.flags & F_ADAPTIVE_WHITES) != 0u) {
        // WhitesTable::for_highlights: the chart's exposure for these highlights.
        float h = clamp(highlights, T[T_WHITES_HIGHLIGHTS], T[T_WHITES_HIGHLIGHTS + 10]);
        int j = 9;
        for (int k = 0; k < 10; ++k)
            if (h <= T[T_WHITES_HIGHLIGHTS + k + 1]) { j = k; break; }
        float w = clamp((h - T[T_WHITES_HIGHLIGHTS + j]) /
                        (T[T_WHITES_HIGHLIGHTS + j + 1] - T[T_WHITES_HIGHLIGHTS + j]), 0.0, 1.0);
        float ev = T[T_WHITES_EXPOSURES + j] + (T[T_WHITES_EXPOSURES + j + 1] - T[T_WHITES_EXPOSURES + j]) * w + 0.24;
        ev = clamp(ev, T[T_WHITES_EXPOSURES], T[T_WHITES_EXPOSURES + 12]);
        int e = 11;
        for (int k = 0; k < 12; ++k)
            if (ev <= T[T_WHITES_EXPOSURES + k + 1]) { e = k; break; }
        float we = clamp((ev - T[T_WHITES_EXPOSURES + e]) /
                         (T[T_WHITES_EXPOSURES + e + 1] - T[T_WHITES_EXPOSURES + e]), 0.0, 1.0);
        int k = row - 3;
        return T[T_WHITES_ADAPTIVE + (e * 3 + k) * 64 + i] * (1.0 - we) +
               T[T_WHITES_ADAPTIVE + ((e + 1) * 3 + k) * 64 + i] * we;
    }
    return T[table + row * 64 + i];
}
float basic_curve(int table, int row, float x) {
    float f = clamp(x * 64.0 - 0.5, -0.5, 63.5);
    int i = clamp(int(floor(f)), 0, 62);
    float a = basic_value(table, row, i), b = basic_value(table, row, i + 1);
    return clamp(a + (b - a) * (f - float(i)), 0.0, 1.0);
}
float basic_slider(int table, float s, float x) {
    if (s == 0.0) return x;
    s = clamp(s, -1.0, 1.0);
    int j = bracket_slot(T_BT_SLIDER, s);
    float y0 = j == 3 ? x : basic_curve(table, j < 3 ? j : j - 1, x);
    float y1 = j + 1 == 3 ? x : basic_curve(table, j + 1 < 3 ? j + 1 : j, x);
    float w = (s - slot_value(T_BT_SLIDER, j)) / (slot_value(T_BT_SLIDER, j + 1) - slot_value(T_BT_SLIDER, j));
    return y0 + (y1 - y0) * w;
}
