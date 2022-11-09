#version 430
layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0) uniform sampler2D input_dens;
layout(binding = 1) uniform writeonly image2D output_fft;


uniform int mesh_exponent;


void main() {
    // This shader handles the index transformation needed for shuffeling the data to apply parallel FFT
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    uvec2 inds = uvec2(pixel_coords);
    ivec2 out_ind = ivec2(0, 0);

    uvec2 bits;
    // the new index is found by reversing the bit order, i didnt know a better way to do that in GLSL
    for (int i=0; i<mesh_exponent; i++) {
        bits = (inds >> i) & 1;
        out_ind += ivec2((uint(1) << (mesh_exponent - i - 1)) * bits);
    }

    float val = texelFetch(input_dens, out_ind, 0).x;
    imageStore(output_fft, pixel_coords, vec4(val, 0.0, 0.0, 0.0));
}