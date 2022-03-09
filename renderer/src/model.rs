use std::f32::consts::PI;
use anyhow::*;
use std::ops::{Mul, Range};
use std::path::Path;
use tobj::LoadOptions;
use uuid::Uuid;
use wgpu::util::DeviceExt;

use crate::{texture};

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub enum MaterialType {
    DIFFUSE,
    EMISSION,
}

pub enum MaterialStructs {
    DIFFUSE(Material), EMISSION(EmissiveMaterial)
}

pub struct Material {
    pub name: String,
    pub mat_type: MaterialType,
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform: MaterialUniform,

}

pub struct EmissiveMaterial {
    pub name: String,
    pub mat_type: MaterialType,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform: LightUniform,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub diffuse_color: [f32; 3],
    pub use_diffuse_color: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub ambient: [f32; 3],
    pub constant: f32,
    pub diffuse: [f32; 3],
    pub linear: f32,
    pub specular: [f32; 3],
    pub quadratic: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshUniform {
    pub position: [f32; 3],
    pub _padding: u32,
    pub worldpos: [f32; 3],
    pub _padding1: u32,
    pub model_offset: [f32; 3],
    pub _padding2: u32,
    pub rot_mat_0: [f32; 3],
    pub _padding3: u32,
    pub rot_mat_1: [f32; 3],
    pub _padding4: u32,
    pub rot_mat_2: [f32; 3],
    pub _padding5: u32,
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub mesh_bind_group: wgpu::BindGroup,
    pub mesh_buffer: wgpu::Buffer,
    pub mesh_uniform: MeshUniform,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Model {
    pub id: Uuid,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<MaterialStructs>,
    pub model_offset: [f32; 3],
    pub model_rot: [f32; 3],
}

impl Model {
    pub fn load<P: AsRef<Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_layout: &wgpu::BindGroupLayout,
        mesh_layout: &wgpu::BindGroupLayout,
        light_layout: &wgpu::BindGroupLayout,
        path: P,
        model_offset: [f32; 3],
        model_rot: [f32; 3],
        id: &String,
    ) -> Result<Self> {
        let obj_uuid =
            Uuid::parse_str(id).expect(&*format!("{:?} doesn't have a valid UUID", path.as_ref()));

        let (obj_models, obj_materials) = tobj::load_obj(
            path.as_ref(),
            &LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )?;
        let obj_materials = obj_materials?;

        // We're assuming that the texture files are stored with the obj file
        let containing_folder = path.as_ref().parent().context("Directory has no parent")?;

        let mut materials: Vec<MaterialStructs> = Vec::new();
        for mat in obj_materials {
            let mat_type;
            let diffuse_texture;
            let bind_group;
            let uniform_buffer;
            let material_uniform;
            match mat.name.as_str() {
                "Ampel.Rot" | "Ampel.Gelb" | "Ampel.Gruen" => {
                    println!("Added light mesh");
                    mat_type = MaterialType::EMISSION;

                    let uniform = LightUniform {
                        ambient: mat.ambient,
                        constant: 1.0,
                        diffuse: mat.diffuse,
                        linear: 0.09,
                        specular: mat.specular,
                        quadratic: 0.032,
                    };

                    uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("LightUniform Buffer"),
                        contents: bytemuck::cast_slice(&[uniform]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                    bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &light_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buffer.as_entire_binding(),
                            }],
                        label: None,
                    });

                    materials.push(MaterialStructs::EMISSION(EmissiveMaterial {
                        name: mat.name,
                        mat_type,
                        bind_group,
                        uniform_buffer,
                        uniform,
                    }));
                },
                _ => {
                    println!("Added diffuse mesh");
                    mat_type = MaterialType::DIFFUSE;
                    let diffuse_path = mat.diffuse_texture;
                    if diffuse_path.is_empty() {
                        diffuse_texture = texture::Texture::load(device, queue, containing_folder.join("solid_white.png"))?;
                        println!("{:?}", mat.diffuse);
                        material_uniform = MaterialUniform {
                            diffuse_color: mat.diffuse,
                            use_diffuse_color: 1,
                        }
                    } else {
                        diffuse_texture =
                        texture::Texture::load(device, queue, containing_folder.join(diffuse_path))?;
                        material_uniform = MaterialUniform {
                            diffuse_color: [1.0, 1.0, 1.0],
                            use_diffuse_color: 0,
                        }
                    }

                    uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("{:?} Uniform Buffer (MATERIAL)", path.as_ref())),
                        contents: bytemuck::cast_slice(&[material_uniform]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                    bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: texture_layout,
                        entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buffer.as_entire_binding(),
                        }
                        ],
                        label: Some(&format!("{:?} Bind Group", path.as_ref())),
                    });
                    materials.push(MaterialStructs::DIFFUSE(Material {
                        name: mat.name,
                        mat_type,
                        diffuse_texture,
                        bind_group,
                        uniform_buffer,
                        uniform: material_uniform,
                    }));

                }
            }


        }

        let mut meshes = Vec::new();
        for m in obj_models {
            let mut vertices = Vec::new();
            let mut x: f32 = 0.0;
            let mut y: f32 = 0.0;
            let mut z: f32 = 0.0;

            for i in 0..m.mesh.positions.len() / 3 {
                x += m.mesh.positions[i * 3];
                y += m.mesh.positions[i * 3 + 1];
                z += m.mesh.positions[i * 3 + 2];

                vertices.push(ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                });
            }

            x = x / (m.mesh.positions.len() / 3) as f32;
            y = y / (m.mesh.positions.len() / 3) as f32;
            z = z / (m.mesh.positions.len() / 3) as f32;

            let mut rot_mat = glam::Mat3::from_rotation_x(model_rot[0] * (PI/180.0));
            rot_mat = rot_mat.mul(glam::Mat3::from_rotation_y(model_rot[1] * (PI/180.0)));
            rot_mat = rot_mat.mul(glam::Mat3::from_rotation_z(model_rot[2] * (PI/180.0)));

            for mut v in vertices.as_mut_slice() {
                v.position = rot_mat.mul(glam::Vec3::from(v.position)).into();
            }


            let uniform = MeshUniform {
                position: [0.0, 0.0, 0.0],
                _padding: 0,
                worldpos: [x, y, z],
                _padding1: 0,
                model_offset: [0.0, 0.0, 0.0],
                _padding2: 0,

                rot_mat_0: rot_mat.x_axis.into(),
                _padding3: 0,
                rot_mat_1: rot_mat.y_axis.into(),
                _padding4: 0,
                rot_mat_2: rot_mat.z_axis.into(),
                _padding5: 0
            };

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", path.as_ref())),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", path.as_ref())),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            let mesh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Uniform Buffer", path.as_ref())),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let mesh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{:?} Mesh bind group", path.as_ref())),
                layout: mesh_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh_buffer.as_entire_binding(),
                }],
            });


            let mesh = Mesh {
                name: m.name,
                vertex_buffer,
                index_buffer,
                mesh_buffer,
                mesh_bind_group,
                mesh_uniform: uniform,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            };
            meshes.push(mesh);

        }

        // let mesh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some(&format!("{:?} Uniform Buffer", path.as_ref())),
        //     contents: bytemuck::cast_slice(&[mesh_uniform]),
        //     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        // });
        //
        // let mesh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     label: Some(&format!("{:?} Model bind group", path.as_ref())),
        //     layout: model_layout,
        //     entries: &[wgpu::BindGroupEntry {
        //         binding: 0,
        //         resource: mesh_buffer.as_entire_binding(),
        //     }],
        // });

        Ok(Self {
            id: obj_uuid,
            meshes,
            materials,
            model_offset,
            model_rot,
        })
    }
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a MaterialStructs,
        camera_bind_group: &'a wgpu::BindGroup,
        light_sources_bind_group: &'a wgpu::BindGroup,
        only_show_emissive: bool,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a MaterialStructs,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_sources_bind_group: &'a wgpu::BindGroup,
        only_show_emissive: bool,
    );

    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        render_pipeline: &'a wgpu::RenderPipeline,
        light_render_pipeline: &'a wgpu::RenderPipeline,
        light_sources_bind_group: &'a wgpu::BindGroup,
        only_show_emissive: bool,
    );
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_sources_bind_group: &'a wgpu::BindGroup,
        render_pipeline: &'a wgpu::RenderPipeline,
        light_render_pipeline: &'a wgpu::RenderPipeline,
        only_show_emissive: bool,
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b MaterialStructs,
        camera_bind_group: &'b wgpu::BindGroup,
        light_sources_bind_group: &'b wgpu::BindGroup,
        only_show_emissive: bool,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group, light_sources_bind_group, only_show_emissive);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b MaterialStructs,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_sources_bind_group: &'b wgpu::BindGroup,
        only_show_emissive: bool,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        match material {
            MaterialStructs::EMISSION(mat) => {
                self.set_bind_group(0, &mat.bind_group, &[]);
                self.set_bind_group(1, camera_bind_group, &[]);
                self.set_bind_group(2, &mesh.mesh_bind_group, &[]);
            },
            MaterialStructs::DIFFUSE(mat) => {
                if only_show_emissive {
                    return;
                }
                self.set_bind_group(0, &mat.bind_group, &[]);
                self.set_bind_group(1, camera_bind_group, &[]);
                self.set_bind_group(2, &mesh.mesh_bind_group, &[]);
                self.set_bind_group(3, light_sources_bind_group, &[]);
            }
        }
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        render_pipeline: &'b wgpu::RenderPipeline,
        light_render_pipeline: &'b wgpu::RenderPipeline,
        light_sources_bind_group: &'b wgpu::BindGroup,
        only_show_emissive: bool,
    ) {
        self.draw_model_instanced(model, 0..1, camera_bind_group, light_sources_bind_group, render_pipeline, light_render_pipeline, only_show_emissive);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_sources_bind_group: &'b wgpu::BindGroup,
        render_pipeline: &'b wgpu::RenderPipeline,
        light_render_pipeline: &'b wgpu::RenderPipeline,
        only_show_emissive: bool,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            match material {
                MaterialStructs::EMISSION(_mat) => {
                    self.set_pipeline(light_render_pipeline) },
                MaterialStructs::DIFFUSE(_mat) => {
                    self.set_pipeline(render_pipeline) },
            }
            self.draw_mesh_instanced(
                mesh,
                material,
                instances.clone(),
                camera_bind_group,
                light_sources_bind_group,
                only_show_emissive,
            );
        }
    }
}