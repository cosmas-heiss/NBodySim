#version 430
#extension GL_NV_shader_atomic_float : enable
layout(local_size_x = 128) in;

layout(std430, binding = 0) coherent buffer input_buffer_kaka {vec4 table[];} body_buffer;
layout(std430, binding = 1) coherent buffer input_buffer_mass {float table[];} input_mass_buffer;
layout(binding = 2) uniform sampler2D input_acc;


uniform float time_step;
uniform mat3x3 fx_ngbhd;
uniform mat3x3 fy_ngbhd;
uniform ivec2 mesh_size;
uniform uint num_bodys;

const float friction_factor = 1.0;
const float eps = 0.001;




ivec2 wrap_coords(ivec2 coords) {
    if (coords.x < 0) {
        coords.x += mesh_size.x;
    }
    if (coords.y < 0) {
        coords.y += mesh_size.y;
    }
    if (coords.x >= mesh_size.x) {
        coords.x -= mesh_size.x;
    }
    if (coords.y >= mesh_size.y) {
        coords.y -= mesh_size.y;
    }
    return coords;
}


vec2 get_surrounding_acc(ivec2 inds, vec2 offset) {
    vec2 acc_out = vec2(0, 0);

    float mass;
    ivec2 mat_inds;
    
    mass = (1.0 - offset.x) * (1.0 - offset.y);
    mat_inds = inds + 1;
    acc_out.x += mass * fx_ngbhd[mat_inds.y][mat_inds.x];
    acc_out.y += mass * fy_ngbhd[mat_inds.y][mat_inds.x];

    mass = offset.x * (1.0 - offset.y);
    mat_inds = inds - ivec2(1, 0) + 1;
    acc_out.x += mass * fx_ngbhd[mat_inds.y][mat_inds.x];
    acc_out.y += mass * fy_ngbhd[mat_inds.y][mat_inds.x];

    mass = (1.0 - offset.x) * offset.y;
    mat_inds = inds - ivec2(0, 1) + 1;
    acc_out.x += mass * fx_ngbhd[mat_inds.y][mat_inds.x];
    acc_out.y += mass * fy_ngbhd[mat_inds.y][mat_inds.x];

    mass = offset.x * offset.y;
    mat_inds = inds - ivec2(1, 1) + 1;
    acc_out.x += mass * fx_ngbhd[mat_inds.y][mat_inds.x];
    acc_out.y += mass * fy_ngbhd[mat_inds.y][mat_inds.x];
    
    return -acc_out;
}


vec2 get_acc(vec2 pos, float mass, inout mat2x2 J_acc) {
    vec2 pixel_pos = pos * vec2(mesh_size) - 0.5;
    vec2 base_pixel = floor(pixel_pos);
    vec2 offset = pixel_pos - base_pixel;
    ivec2 base_pixel_int = ivec2(base_pixel);

    // the acceleration is bilinearly interpolated between neighbouring pixels
    // for each fetch, we need to subtract the auto-gravitation of the body
    vec2 acc1 = texelFetch(input_acc, wrap_coords(base_pixel_int), 0).xy;
    acc1 -= mass * get_surrounding_acc(ivec2(0, 0), offset);

    vec2 acc2 = texelFetch(input_acc, wrap_coords(base_pixel_int + ivec2(1, 0)), 0).xy;
    acc2 -= mass * get_surrounding_acc(ivec2(1, 0), offset);
    
    vec2 acc3 = texelFetch(input_acc, wrap_coords(base_pixel_int + ivec2(0, 1)), 0).xy;
    acc3 -= mass * get_surrounding_acc(ivec2(0, 1), offset);

    vec2 acc4 = texelFetch(input_acc, wrap_coords(base_pixel_int + ivec2(1, 1)), 0).xy;
    acc4 -= mass * get_surrounding_acc(ivec2(1, 1), offset);

    vec2 acc = mix(mix(acc1, acc2, offset.x), mix(acc3, acc4, offset.x), offset.y);

    J_acc[0] = mix(acc2 - acc1, acc4 - acc3, offset.y);
    J_acc[1] = mix(acc3 - acc1, acc4 - acc2, offset.x);

    return acc;
}



void main() {
    // this program does the stepping procedure
    uint buffer_index = uint(gl_GlobalInvocationID.x);
    if (buffer_index < num_bodys) {

    vec4 pos_vel = body_buffer.table[buffer_index];
    float mass = input_mass_buffer.table[buffer_index];
    vec2 original_pos = pos_vel.xy;


    vec2 pixel_pos = original_pos * vec2(mesh_size) - 0.5;
    vec2 base_pixel = floor(pixel_pos);
    vec2 offset = pixel_pos - base_pixel;


    vec2 pos = pos_vel.xy;
    vec2 vel = pos_vel.zw;

    // getting acceleration and jacobian of acceleretaion
    mat2x2 J_acc;
    vec2 acc = get_acc(pos, mass, J_acc);


    // frobeniusnorm factor to make sure that I - 0.5 dt * J_acc is invertible
    // The stepping is implicit with the first derivative as an estimation of the value at the target point
    float J_acc_frob_norm_factor = 1.0 / max(sqrt(J_acc[0].x * J_acc[0].x + J_acc[0].y * J_acc[0].y + J_acc[1].x * J_acc[1].x + J_acc[1].y * J_acc[1].y), 1.0);
    vec2 new_vel = inverse(mat2x2(1.0) - 0.5 * J_acc_frob_norm_factor * time_step * transpose(J_acc)) * (vel + time_step * acc);
    pos += time_step * new_vel;

    // this wraps the position inside the [0,1] square
    pos -= floor(pos);
    body_buffer.table[buffer_index] = vec4(pos, new_vel * friction_factor);
    }
}