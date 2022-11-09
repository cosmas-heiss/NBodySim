#version 430
layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0) uniform sampler2D input_fft;
layout(binding = 1) uniform sampler2D input_factor;
layout(binding = 2) uniform writeonly image2D output_fft;


uniform int mesh_exponent;


void main() {
    // this shader does the complex multiplication with the force fft filter and reshuffles the entries
    // for the next fft pass to go back to the spatial domain
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    uvec2 inds = uvec2(pixel_coords);
    ivec2 out_ind = ivec2(0, 0);

    uvec2 bits;
    // bit shuffle shit, see FFTInputShader for more details
    for (int i=0; i<mesh_exponent; i++) {
        bits = (inds >> i) & 1;
        out_ind += ivec2((uint(1) << (mesh_exponent - i - 1)) * bits);
    }

    vec2 fft_val = texelFetch(input_fft, out_ind, 0).xy / float(uint(1) << (2 * mesh_exponent));
    vec2 fs = texelFetch(input_factor, out_ind, 0).xy;

    // complex mult, such that we get
    // (real(fft_val * i * fs.x), imag(fft_val * i * fs.x), real(fft_val * i * fs.y), imag(fft_val * i * fs.y))
    vec4 val = vec4(-fft_val.y * fs.x, fft_val.x * fs.x, -fft_val.y * fs.y, fft_val.x * fs.y);

    imageStore(output_fft, pixel_coords, val);
}