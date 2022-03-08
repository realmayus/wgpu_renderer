// Vertex shader

[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

struct MaterialUniform {
    diffuse_color: vec3<f32>;
    use_diffuse_color: i32;
};
[[group(0), binding(2)]]
var<uniform> material_uniform: MaterialUniform;

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
    [[location(1)]] tex_coords: vec2<f32>;
    [[location(2)]] normal: vec3<f32>;
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
    [[location(0)]] tex_coords: vec2<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] world_position: vec3<f32>;
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
    out.tex_coords = model.tex_coords;
    out.world_normal = normal_matrix * model.normal;
    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position + model_uniform.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    return out;
}

// Fragment shader



struct LightUniform {
    ambient: vec3<f32>;
    diffuse: vec3<f32>;
    specular: vec3<f32>;
    constant: f32;
    linear: f32;
    quadratic: f32;
};

//struct SceneInformation {
//    light_source_count: i32;
//};
//[[group(3), binding(0)]]
//var<uniform> scene_info: SceneInformation;

struct LightSource {
    ambient: vec3<f32>;
    constant: f32;
    diffuse: vec3<f32>;
    linear: f32;
    specular: vec3<f32>;
    quadratic: f32;
    position: vec3<f32>;
    worldpos: vec3<f32>;
    //light_uniform: LightUniform;
    //mesh_uniform: ModelUniform;
};

struct LightSourceArray {
    arr: array<LightSource>;
};

[[group(3), binding(0)]]
var<storage, read> lights: LightSourceArray;

fn calculate_lights(light: LightSource, in: VertexOutput, obj_color: vec3<f32>) -> vec3<f32> {
    let lightDir: vec3<f32> = normalize((light.position + light.worldpos) - in.world_position);
    // diffuse shading
    let diff: f32 = max(dot(in.world_normal, lightDir), 0.0);
    // specular shading
    //let reflectDir: vec3<f32> = reflect(-lightDir, in.world_normal);
    let reflectDir: vec3<f32> = reflect(-lightDir, in.world_normal);
    let spec: f32 = pow(max(dot(in.world_normal, reflectDir), 0.0), 32.0);
    // attenuation
    let distance: f32 = length((light.position + light.worldpos) - in.world_position);
    let attenuation: f32 = 1.0 / (light.constant + light.linear * distance +
                 light.quadratic * (distance * distance));
    // combine results
    var ambient: vec3<f32> = light.ambient * obj_color;
    var diffuse: vec3<f32> = light.diffuse * diff * obj_color;
    var specular: vec3<f32> = light.specular * spec * obj_color;
    ambient = ambient * attenuation;
    diffuse = diffuse * attenuation;
    specular = specular * attenuation;

    return ambient + diffuse + specular;
}

//fn calc_light_simple(in: VertexOutput, color_: vec3<f32>) -> vec3<f32> {
//    var i: i32 = 0;
//    var color: vec3<f32> = vec3<f32>(color_[0], color_[1], color_[2]);
//
//    var ambient_color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
//    var diffuse_color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
//    var specular_color: vec3<f32>  = vec3<f32>(0.0, 0.0, 0.0);
//
//    loop {
//        if (i >= 1) {
//            break;
//        }
//        var light = lights.arr[i];
//        // We don't need (or want) much ambient light, so 0.1 is fine
//        let ambient_strength = 0.1;
//        ambient_color = ambient_color + light.diffuse * ambient_strength;
//
//        let light_dir = normalize(light.position - in.world_position);
//        let view_dir = normalize(camera.view_pos.xyz - in.world_position);
//        let half_dir = normalize(view_dir + light_dir);
//
//        let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
//        diffuse_color = diffuse_color + light.diffuse * diffuse_strength;
//
//        let specular_strength = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0);
//        specular_color = specular_color + specular_strength * light.diffuse;
//
//
//        continuing {
//            i = i + 1;
//        }
//    }
//    return (ambient_color + diffuse_color + specular_color) * color;
//}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    // We don't need (or want) much ambient light, so 0.1 is fine
    //let ambient_strength = 0.1;
    //let ambient_color = light_material_uniform.diffuse_color * ambient_strength;

    //let light_dir = normalize(light_mesh_uniform.position - in.world_position);
    //let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    //let half_dir = normalize(view_dir + light_dir);

    //let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    //let diffuse_color = light_material_uniform.diffuse_color * diffuse_strength;

    //let specular_strength = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0);
    //let specular_color = specular_strength * light_material_uniform.diffuse_color;

    var result = calculate_lights(lights.arr[0], in, material_uniform.diffuse_color);

    for (var i: i32 = 1; i < 6; i = i + 1) {
        result = result + calculate_lights(lights.arr[i], in, material_uniform.diffuse_color);
    }

    //var result: vec3<f32>;
    //if (material_uniform.use_diffuse_color == 0) {
    //    result = calc_light_simple(in, object_color.xyz);
    //    //let result = object_color.xyz;
    //} else if (material_uniform.use_diffuse_color == 1) {
    //    result = calc_light_simple(in, material_uniform.diffuse_color);
    //    //let result = material_uniform.diffuse_color;
    //}

    return vec4<f32>(result, object_color.a);
}

struct LightResult {
    ambient_color: vec3<f32>;
    diffuse_color: vec3<f32>;
    specular_color: vec3<f32>;
};

