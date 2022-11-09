#version 430
layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0) uniform sampler2D input_fft;
layout(binding = 1) uniform writeonly image2D output_acc;


uniform ivec2 mesh_size;

void main() {
    // This shader does a transformation f on the input such that f(dft(dft(a))) = idft(dft(a))
    // The transformation is just a mirroring and offset by one pixel
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    vec4 fft_vals = texelFetch(input_fft, pixel_coords, 0);

    ivec2 ind = mesh_size - pixel_coords;
    if (ind.x == mesh_size.x) {
        ind.x = 0;
    }
    if (ind.y == mesh_size.y) {
        ind.y = 0;
    }

    imageStore(output_acc, ind, vec4(fft_vals.xz, 0.0, 0.0));
}