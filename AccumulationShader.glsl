#version 430
#extension GL_NV_shader_atomic_float : enable
layout(local_size_x = 128) in;

layout(std430, binding = 0) coherent buffer input_buffer_kaka {vec4 table[];} input_body_buffer;
layout(std430, binding = 1) coherent buffer input_buffer_mass {float table[];} input_mass_buffer;

layout(r32f, binding = 2) uniform image2D output_accumulation;


uniform ivec2 mesh_size;
uniform uint num_bodys;


ivec2 periodic_index(ivec2 ind) {
    // wraps the coords in a periodic fashion.. not really nice because of not using mod,
    // but mod is implemented for floats...
    if (ind.x < 0) {
        ind.x += mesh_size.x;
    }
    if (ind.x >= mesh_size.x) {
        ind.x -= mesh_size.x;
    }
    if (ind.y < 0) {
        ind.y += mesh_size.y;
    }
    if (ind.y >= mesh_size.y) {
        ind.y -= mesh_size.y;
    }
    return ind;
}

void main() {
    // This shader takes all the bodies from the input buffer and writes their mass contribution to the
    // density distribution texture with bilinear interpolation
    uint buffer_index = uint(gl_GlobalInvocationID.x);
    // I do this line if the number of bodies is not exactly divisible by the group size
    // probably bad practise but it works...
    if (buffer_index < num_bodys) {

    // get position and mass information
    vec2 pos = input_body_buffer.table[buffer_index].xy;
    float mass = input_mass_buffer.table[buffer_index];

    // do the bilinear stuff and add to surrounding pixels
    vec2 pixel_pos = pos * vec2(mesh_size) - 0.5;
    vec2 base_pixel = floor(pixel_pos);
    vec2 offset = pixel_pos - base_pixel;
    ivec2 ibase_pixel = ivec2(base_pixel);

    imageAtomicAdd(output_accumulation, periodic_index(ibase_pixel), mass * (1.0 - offset.x) * (1.0 - offset.y));
    imageAtomicAdd(output_accumulation, periodic_index(ibase_pixel + ivec2(1, 0)), mass * offset.x * (1.0 - offset.y));
    imageAtomicAdd(output_accumulation, periodic_index(ibase_pixel + ivec2(0, 1)), mass * (1.0 - offset.x) * offset.y);
    imageAtomicAdd(output_accumulation, periodic_index(ibase_pixel + ivec2(1, 1)), mass * offset.x * offset.y);
    }
}