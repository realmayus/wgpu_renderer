struct LightUniform {
    ambient: vec3<f32>;
    constant: f32;
    diffuse: vec3<f32>;
    linear: f32;
    specular: vec3<f32>;
    quadratic: f32;
};
[[group(0), binding(0)]]
var<uniform> light_uniform: LightUniform;


struct Camera {
    view_pos: vec4<f32>;
    view_proj: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> camera: Camera;

struct ModelUniform {
    position: vec3<f32>;
};
[[group(2), binding(0)]]
var<uniform> model_uniform: ModelUniform;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
};

struct InstanceInput {
    [[location(5)]] model_matrix_0: vec4<f32>;
    [[location(6)]] model_matrix_1: vec4<f32>;
    [[location(7)]] model_matrix_2: vec4<f32>;
    [[location(8)]] model_matrix_3: vec4<f32>;
    [[location(9)]] normal_matrix_0: vec3<f32>;
    [[location(10)]] normal_matrix_1: vec3<f32>;
    [[location(11)]] normal_matrix_2: vec3<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] color: vec3<f32>;
};

[[stage(vertex)]]
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );
    var out: VertexOutput;
    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position + model_uniform.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.color = light_uniform.diffuse;
    return out;

}

// Fragment shader

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}