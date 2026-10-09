#version 450
// Draws the photo's rectangle as a 4-vertex triangle strip; no vertex buffer.

layout(push_constant) uniform Params {
    vec4 rect;      // photo rectangle in NDC: x0, y0, x1, y1
    float exposure; // linear gain, 2^EV
    float contrast; // S-curve exponent, 1 = unchanged
} pc;

layout(location = 0) out vec2 uv;

void main() {
    vec2 corner = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
    uv = corner;
    gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, corner), 0.0, 1.0);
}
