#version 430
#extension GL_NV_shader_atomic_float : enable
layout(local_size_x = 128) in;

layout(std430, binding = 0) coherent buffer input_buffer_kaka {vec4 table[];} input_body_buffer;
layout(std430, binding = 1) coherent buffer input_buffer_mass {float table[];} input_mass_buffer;

layout(r32f, binding = 2) uniform image2D output_acc;


uniform uvec2 size;
uniform uint num_bodys;
uniform vec4 bbox;

const float decay = 0.7;
const int num_steps = 0;
const float base_length = 0.005;


void rasterize(float mass, vec2 pos, vec2 vel) {
    float signX = 1.0;
    float signY = 1.0;

    if (vel.x < 0) {
        signX = -1.0;
    }
    if (vel.y < 0) {
        signY = -1.0;
    }

    float vel_factor = min(length(vel) / float(num_steps) * decay, 0.9);
    vel = abs(vel);

    if (vel.y <= vel.x) {
        float m = vel.y / vel.x;
        for (int i=0; i<num_steps; i++) {
            imageAtomicAdd(output_acc, ivec2(floor(pos.x * size.x + signX * float(i)), floor(pos.y * size.y + signY * m * float(i))), mass);
            mass *= vel_factor;
            if (mass < 0.1) {
                break;
            }
        }
    } else {
        float m = vel.x / vel.y;
        for (int i=0; i<num_steps; i++) {
            imageAtomicAdd(output_acc, ivec2(floor(pos.x * size.x + float(i) * m * signX), floor(pos.y * size.y + float(i) * signY)), mass);
            mass *= vel_factor;
            if (mass < 0.1) {
                break;
            }
        }
    }
}


void main() {
    uint buffer_index = uint(gl_GlobalInvocationID.x);
    if (buffer_index < num_bodys) {
    vec4 pos_vel = input_body_buffer.table[buffer_index];
    vec2 pos = pos_vel.xy;
    vec2 vel = pos_vel.zw;

    pos.x = (pos.x - bbox.x) / (bbox.y - bbox.x);
    pos.y = (pos.y - bbox.z) / (bbox.w - bbox.z);

    // only the bodies inside viewport are drawn, the rasterize is to give them tails, but thats a bit expensive
    if (pos.x * size.x >= -num_steps && pos.x * size.x < size.x + num_steps && pos.y * size.y >= -num_steps && pos.y * size.y < size.y + num_steps) {
        float mass = input_mass_buffer.table[buffer_index];

        //rasterize(mass, pos, -vel * base_length / vec2(bbox.y - bbox.x, bbox.w - bbox.z));

        imageAtomicAdd(output_acc, ivec2(floor(pos * size)), mass);
        }
    }
}
