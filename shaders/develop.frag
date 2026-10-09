#version 450
#extension GL_GOOGLE_include_directive : require
// The per-pixel stage, once per screen pixel: tone stage at the Exposure, the Shadows
// and Highlights gain from the map (local_tone.rs gain), the Basic curve as DNG
// RGBTone in encoded ProPhoto (tone.rs curve_input), then clipped sRGB.
#define FRAGMENT_STAGE
#include "common.glsl"

layout(set = 0, binding = 6) uniform sampler2D photo; // on-screen size, linear sRGB
// With the cache: the same pixels after the tone stage at the current Exposure.
layout(set = 0, binding = 7) uniform sampler2D toned;
layout(constant_id = 0) const bool TONED_INPUT = false;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

// The map at this position, bilinear as LocalToneMap::bilinear reads it.
float map_at(uint offset, vec2 at) {
    float fx = clamp(at.x * float(pc.w) - 0.5, 0.0, float(pc.w - 1u));
    float fy = clamp(at.y * float(pc.h) - 0.5, 0.0, float(pc.h - 1u));
    uint ix = uint(fx), iy = uint(fy);
    uint jx = min(ix + 1u, pc.w - 1u), jy = min(iy + 1u, pc.h - 1u);
    float tx = fx - float(ix), ty = fy - float(iy);
    float top = S[offset + iy * pc.w + ix] * (1.0 - tx) + S[offset + iy * pc.w + jx] * tx;
    float bottom = S[offset + jy * pc.w + ix] * (1.0 - tx) + S[offset + jy * pc.w + jx] * tx;
    return top * (1.0 - ty) + bottom * ty;
}

float lut_eval(float x) {
    float f = clamp(x, 0.0, 1.0) * 1024.0;
    uint i = min(uint(f), 1023u);
    return lut[i] + (lut[i + 1u] - lut[i]) * (f - float(i));
}

void main() {
    vec3 rgb = TONED_INPUT ? texture(toned, uv).rgb : tone_stage(texture(photo, uv).rgb, pc.gain, pc.black);
    if ((pc.flags & F_MAP) != 0u) {
        float base = map_at(S_FA(), uv) * log2(max(luminance(rgb), 6e-4)) + map_at(S_FB(), uv);
        float ev = (pc.shadows_s != 0.0 ? curve_eval(T_SHADOWS, key_shadows, base, true) : 0.0) +
                   (pc.highlights_s != 0.0 ? curve_eval(T_HIGHLIGHTS, key_highlights, base, false) : 0.0);
        rgb *= exp2(ev);
    }
    vec3 p = clamp(mulr(RGB_TO_PRO, rgb), 0.0, 1.0);
    p = vec3(srgb_encode(p.r), srgb_encode(p.g), srgb_encode(p.b));
    if ((pc.flags & F_BASIC) != 0u) {
        float lo = min(min(p.r, p.g), p.b);
        float hi = max(max(max(p.r, p.g), p.b), 0.0);
        p = rgb_tone(p, lut_eval(lo), lut_eval(hi));
    }
    p = clamp(p, 0.0, 1.0);
    vec3 lin = clamp(mulr(PRO_TO_RGB, vec3(srgb_decode(p.r), srgb_decode(p.g), srgb_decode(p.b))), 0.0, 1.0);
    color = vec4(srgb_encode(lin.r), srgb_encode(lin.g), srgb_encode(lin.b), 1.0);
}
