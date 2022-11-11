#version 430
layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0) uniform sampler2D input_colormap;
layout(binding = 1) uniform sampler2D input_vals;
layout(binding = 2) uniform writeonly uimage2D output_tex;


uniform ivec2 size;
uniform float color_norm_factor;


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    if (pixel_coords.x < size.x && pixel_coords.y < size.y) {

    float val = texelFetch(input_vals, pixel_coords, 0).x;
    
    if (val < 100.0) {
        val = val * 0.01;
    } else {
        val = log(val) - 3.60517019;
    }
    val = min(val / color_norm_factor, 1.0);

    vec4 color = texture(input_colormap, vec2(clamp(val, 0.015625, 0.984375), 0.0));

    imageStore(output_tex, pixel_coords, uvec4(color));
    }
}