#version 430
layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0) uniform sampler2D input_fft;
layout(binding = 1) uniform writeonly image2D output_fft;


uniform bool x_or_y;
uniform int pow_two_level;

const float pi = 3.14159265359;

void main() {
    // This shader performs one step of the FFT algorithm with a block-size pow_two_level
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    // this is done to be able to use this program for the x or y direction respectively
    int ind_in_question;
    if (x_or_y) {
        ind_in_question = pixel_coords.x;
    } else {
        ind_in_question = pixel_coords.y;
    }

    // divide the input into blocks which are handled each individually in some way
    int base_block_index = pow_two_level * (ind_in_question / pow_two_level);
    int block_index = ind_in_question - base_block_index;
    int E_index = block_index - (pow_two_level / 2) * (block_index / (pow_two_level / 2));
    int O_index = E_index + (pow_two_level / 2);

    // do the twiddle factor
    vec2 twiddle_factor = vec2(cos(-2.0 * pi * float(E_index) / float(pow_two_level)), sin(-2.0 * pi * float(E_index) / float(pow_two_level)));
    if (block_index >= pow_two_level / 2) {
        twiddle_factor *= -1;
    }

    ivec2 getting_index_E;
    ivec2 getting_index_O;
    if (x_or_y) {
        getting_index_E = ivec2(E_index + base_block_index, pixel_coords.y);
        getting_index_O = ivec2(O_index + base_block_index, pixel_coords.y);
    } else {
        getting_index_E = ivec2(pixel_coords.x, E_index + base_block_index);
        getting_index_O = ivec2(pixel_coords.x, O_index + base_block_index);
    }
    
    vec4 E_val = texelFetch(input_fft, getting_index_E, 0);
    vec4 O_val = texelFetch(input_fft, getting_index_O, 0);
    O_val = vec4(O_val.xz * twiddle_factor.x - O_val.yw * twiddle_factor.y, O_val.xz * twiddle_factor.y + O_val.yw * twiddle_factor.x).xzyw;
    imageStore(output_fft, pixel_coords, vec4(E_val + O_val));
}