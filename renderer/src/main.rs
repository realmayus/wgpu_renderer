extern crate core;

use std::ops::Add;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

use cgmath::prelude::*;
use egui::FontDefinitions;
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::{Platform, PlatformDescriptor};
use epi::*;
use log::{debug, log};
use uuid::Uuid;
use wgpu::util::DeviceExt;
use wgpu::BufferSize;
use winit::dpi::{LogicalSize, Size};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::ampel::AmpelStatus;
use crate::camera::{Camera, CameraController};
use crate::model::{DrawModel, LightUniform, MaterialStructs, MeshUniform, Model, Vertex};
use crate::world::{SceneItem, World};

mod ampel;
mod camera;
mod model;
mod texture;
mod util;
mod world;

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let model =
            cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            model: model.into(),
            // NEW!
            normal: cgmath::Matrix3::from(self.rotation).into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl model::Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We don't have to do this in code though.
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // NEW!
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CompareFunction {
    Undefined = 0,
    Never = 1,
    Less = 2,
    Equal = 3,
    LessEqual = 4,
    Greater = 5,
    NotEqual = 6,
    GreaterEqual = 7,
    Always = 8,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    index: i32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(&shader);

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(format!("Render Pipeline {}", index).as_str()),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    println!("After render pipeline");

    pipeline
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneInformation {
    pub light_source_count: i32,
}

pub struct LightSource<'a> {
    pub light_uniform: &'a LightUniform,
    pub buffer: &'a wgpu::Buffer,
    pub mesh_uniform: &'a MeshUniform,
    pub mesh_buffer: &'a wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightSourcePod {
    pub ambient: [f32; 3],
    pub constant: f32,
    pub diffuse: [f32; 3],
    pub linear: f32,
    pub specular: [f32; 3],
    pub quadratic: f32,
    pub position: [f32; 3],
    pub _padding: u32,
    pub worldpos: [f32; 3],
    pub _padding2: u32,
    pub model_offset: [f32; 3],
    pub _padding3: u32,
    pub rot_mat_0: [f32; 3],
    pub _padding4: u32,
    pub rot_mat_1: [f32; 3],
    pub _padding5: u32,
    pub rot_mat_2: [f32; 3],
    pub _padding6: u32,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    camera: camera::Camera,
    camera_controller: camera::CameraController,
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    models: Vec<Model>,
    light_render_pipeline: wgpu::RenderPipeline,
    platform: Platform,
    egui_rpass: RenderPass,
    surface_config: wgpu::SurfaceConfiguration,
    window: Window,
    clear_color: [f32; 3],
    light_model: Model,
    world: World,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    simple_uniform_layout: wgpu::BindGroupLayout,
    light_sources_buffer: wgpu::Buffer,
    light_sources_bind_group: wgpu::BindGroup,
    only_show_emissive: bool,
    ampel_index: i32,
    next_cycle: std::time::Duration,
    green_duration: u32,
    yellow_duration: u32,
    red_yellow_duration: u32,
    all_red_duration: u32,
}

impl State {
    async fn new(window: Window) -> State {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(&window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_format = surface.get_preferred_format(&adapter).unwrap();

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &surface_config);

        //------- EGUI

        // We use the egui_wgpu_backend crate as the render backend.
        let egui_rpass = RenderPass::new(&device, surface_format, 1);

        let platform = Platform::new(PlatformDescriptor {
            physical_width: size.width as u32,
            physical_height: size.height as u32,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });
        //--------

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        // Sampled Texture
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // Sampler
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let camera_bind_group_layout = Camera::create_bind_group_layout(&device);
        let camera = Camera::new(&surface_config, &device, &camera_bind_group_layout);

        let camera_controller = CameraController::new(0.8);

        const NUM_INSTANCES_PER_ROW: i32 = 1;
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|_| {
                (0..NUM_INSTANCES_PER_ROW).map(move |_| {
                    let position = cgmath::Vector3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    };

                    Instance {
                        position,
                        rotation: cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        ),
                    }
                })
            })
            .collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &surface_config, "depth_texture");

        let simple_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("simple_uniform_layout"),
            });

        let simple_storage_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(144),
                    },
                    count: None,
                }],
                label: Some("simple_storage_layout"),
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &simple_uniform_layout,
                    &simple_storage_layout,
                ],
                push_constant_ranges: &[],
            });
        println!("Before initial pipeline creation");
        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                0,
            )
        };
        println!("After initial pipeline creation");
        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[
                    &simple_uniform_layout,
                    &camera_bind_group_layout,
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                1,
            )
        };

        let res_dir = env::current_dir().unwrap().join("renderer/res");

        let mut light_sources: Vec<LightSourcePod> = vec![];

        let (world, mut models) = State::load_world(
            &device,
            &queue,
            &texture_bind_group_layout,
            &simple_uniform_layout,
            &simple_uniform_layout,
        );

        let light_model = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            &simple_uniform_layout,
            &simple_uniform_layout,
            res_dir.join("cube.obj"),
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            &String::from("6b761f1a-c88c-436c-8834-494a541e084c"),
        )
        .unwrap();

        log!(log::Level::Warn, "Loaded {} models.", &models.len());

        for model in models.as_mut_slice() {
            for m in model.meshes.as_mut_slice() {
                // Populate light_sources
                let mat_structs = &model.materials[m.material];
                if m.material != 0 {
                    //TODO check if this works
                    match mat_structs {
                        MaterialStructs::EMISSION(mat) => {
                            light_sources.push(LightSourcePod {
                                ambient: mat.uniform.ambient,
                                constant: mat.uniform.constant,
                                diffuse: mat.uniform.diffuse,
                                linear: mat.uniform.linear,
                                specular: mat.uniform.specular,
                                quadratic: mat.uniform.quadratic,
                                position: m.mesh_uniform.position,
                                _padding: 0,
                                worldpos: m.mesh_uniform.worldpos,
                                _padding2: 0,
                                model_offset: m.mesh_uniform.model_offset,
                                _padding3: 0,
                                rot_mat_0: m.mesh_uniform.rot_mat_0,
                                _padding4: 0,
                                rot_mat_1: m.mesh_uniform.rot_mat_1,
                                _padding5: 0,
                                rot_mat_2: m.mesh_uniform.rot_mat_2,
                                _padding6: 0,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        let light_sources_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light sources buffer"),
            contents: bytemuck::cast_slice(light_sources.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let light_sources_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &simple_storage_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_sources_buffer.as_entire_binding(),
            }],
            label: None,
        });

        Self {
            surface,
            device,
            queue,
            surface_config,
            size,
            render_pipeline,
            camera,
            camera_controller,
            instance_buffer,
            depth_texture,
            models,
            light_render_pipeline,
            platform,
            egui_rpass,
            window,
            clear_color: [0.0, 0.0, 0.0],
            light_model,
            world,
            texture_bind_group_layout,
            simple_uniform_layout,
            light_sources_buffer,
            light_sources_bind_group,
            only_show_emissive: false,
            ampel_index: 0,
            next_cycle: std::time::Duration::new(0, 0),
            green_duration: 10,
            yellow_duration: 5,
            red_yellow_duration: 5,
            all_red_duration: 5,
        }
    }

    fn load_world(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        model_bind_group_layout: &wgpu::BindGroupLayout,
        light_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> (World, Vec<Model>) {
        let res_dir = env::current_dir().unwrap().join("renderer/res");
        let world: World = toml::from_str(
            &*fs::read_to_string(res_dir.join("world.toml")).expect("Couldn't open world.toml"),
        )
        .expect("Couldn't deserialize world.toml");

        let mut models = vec![];
        for m in world.scene.as_slice() {
            models.push(
                model::Model::load(
                    device,
                    queue,
                    texture_bind_group_layout,
                    model_bind_group_layout,
                    light_bind_group_layout,
                    res_dir.join(&m.model_file),
                    m.position,
                    m.rotation,
                    &m.id,
                )
                .expect(&*format!("Couldn't load model {}", &m.model_file)),
            );
        }

        (world, models)
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
        self.depth_texture = texture::Texture::create_depth_texture(
            &self.device,
            &self.surface_config,
            "depth_texture",
        );
        self.camera
            .update_aspect(new_size.width as f32, new_size.height as f32);
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn find_model(models: &mut [Model], id: String) -> &mut Model {
        models
            .iter_mut()
            .filter(|m| m.id == Uuid::parse_str(&*id).unwrap())
            .next()
            .unwrap()
    }

    // fn setAmpel()

    fn update(&mut self) {
        self.camera.update_view_proj();
        self.camera_controller.update_camera(&mut self.camera);
        self.queue.write_buffer(
            &self.camera.buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform]),
        );

        self.queue.write_buffer(
            &self.light_model.meshes[0].mesh_buffer,
            0,
            bytemuck::cast_slice(&[self.light_model.meshes[0].mesh_uniform]),
        );

        if self.next_cycle > std::time::Duration::new(0, 0)
            && SystemTime::now().duration_since(UNIX_EPOCH).unwrap() > self.next_cycle
        {
            match self.ampel_index {
                0 => {
                    // All red
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    self.ampel_index = 1;
                }
                1 => {
                    // red-yellow - red - red-yellow - red
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::REDYELLOW);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::REDYELLOW);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);

                    self.ampel_index = 2;
                    self.next_cycle = SystemTime::now()
                        .add(std::time::Duration::new(self.red_yellow_duration as u64, 0))
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                }
                2 => {
                    // green - red - green - red
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::GREEN);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::GREEN);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);

                    self.ampel_index = 3;
                    self.next_cycle = SystemTime::now()
                        .add(std::time::Duration::new(self.green_duration as u64, 0))
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                }
                3 => {
                    // yellow - red - yellow - red
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::YELLOW);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::YELLOW);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);

                    self.ampel_index = 4;
                    self.next_cycle = SystemTime::now()
                        .add(std::time::Duration::new(self.yellow_duration as u64, 0))
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                }
                4 => {
                    // red - red - red - red
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);

                    self.ampel_index = 5;
                    self.next_cycle = SystemTime::now()
                        .add(std::time::Duration::new(self.all_red_duration as u64, 0))
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                }
                5 => {
                    // red - red-yellow - red - red-yellow
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::REDYELLOW);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::REDYELLOW);

                    self.ampel_index = 6;
                    self.next_cycle = SystemTime::now()
                        .add(std::time::Duration::new(self.red_yellow_duration as u64, 0))
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                }
                6 => {
                    // red - green - red - green
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::GREEN);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::GREEN);

                    self.ampel_index = 7;
                    self.next_cycle = SystemTime::now()
                        .add(std::time::Duration::new(self.green_duration as u64, 0))
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                }
                7 => {
                    // red - yellow - red - yellow
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::YELLOW);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::YELLOW);

                    self.ampel_index = 8;
                    self.next_cycle = SystemTime::now()
                        .add(std::time::Duration::new(self.yellow_duration as u64, 0))
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                }
                8 => {
                    // red - red - red - red
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "5a49699e-b09d-46bd-bafb-0463fcef5054".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "2be7a064-9ee4-4097-9797-d6d693595b3c".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);
                    ampel::Ampel::from_model(State::find_model(
                        self.models.as_mut_slice(),
                        "c67f737a-b2ad-43ad-ad21-99c87735a141".to_string(),
                    ))
                    .unwrap()
                    .set_status(AmpelStatus::RED);

                    self.ampel_index = 1;
                    self.next_cycle = SystemTime::now()
                        .add(std::time::Duration::new(self.all_red_duration as u64, 0))
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                }
                _ => {}
            }
        }

        match &self.light_model.materials[0] {
            MaterialStructs::EMISSION(mat) => {
                self.queue.write_buffer(
                    &mat.uniform_buffer,
                    0,
                    bytemuck::cast_slice(&[mat.uniform]),
                );
            }
            _ => {}
        }
        let mut light_sources: Vec<LightSourcePod> = vec![];
        for model in self.models.as_mut_slice() {
            for m in model.meshes.as_mut_slice() {
                m.mesh_uniform.model_offset = model.model_offset;
                self.queue
                    .write_buffer(&m.mesh_buffer, 0, bytemuck::cast_slice(&[m.mesh_uniform]));

                // Populate light_sources
                let mat_structs = &model.materials[m.material];
                match mat_structs {
                    MaterialStructs::EMISSION(mat) => {
                        light_sources.push(LightSourcePod {
                            ambient: mat.uniform.ambient,
                            constant: mat.uniform.constant,
                            diffuse: mat.uniform.diffuse,
                            linear: mat.uniform.linear,
                            specular: mat.uniform.specular,
                            quadratic: mat.uniform.quadratic,
                            position: m.mesh_uniform.position,
                            _padding: 0,
                            worldpos: m.mesh_uniform.worldpos,
                            _padding2: 0,
                            model_offset: m.mesh_uniform.model_offset,
                            _padding3: 0,
                            rot_mat_0: m.mesh_uniform.rot_mat_0,
                            _padding4: 0,
                            rot_mat_1: m.mesh_uniform.rot_mat_1,
                            _padding5: 0,
                            rot_mat_2: m.mesh_uniform.rot_mat_2,
                            _padding6: 0,
                        });
                    }
                    _ => {}
                }
            }

            self.queue.write_buffer(
                &self.light_sources_buffer,
                0,
                bytemuck::cast_slice(&light_sources),
            );

            for mat_ in model.materials.as_slice() {
                match mat_ {
                    MaterialStructs::EMISSION(mat) => {
                        self.queue.write_buffer(
                            &mat.uniform_buffer,
                            0,
                            bytemuck::cast_slice(&[mat.uniform]),
                        );
                    }
                    MaterialStructs::DIFFUSE(mat) => {
                        self.queue.write_buffer(
                            &mat.uniform_buffer,
                            0,
                            bytemuck::cast_slice(&[mat.uniform]),
                        );
                    }
                }
            }
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        {
            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: self.clear_color[0] as f64,
                                g: self.clear_color[1] as f64,
                                b: self.clear_color[2] as f64,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });

                render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

                for m in self.models.as_slice() {
                    render_pass.draw_model(
                        &m,
                        &self.camera.bind_group,
                        &self.render_pipeline,
                        &self.light_render_pipeline,
                        &self.light_sources_bind_group,
                        self.only_show_emissive,
                    );
                }
            }

            self.queue.submit(std::iter::once(encoder.finish()));

            self.platform.begin_frame();

            self.draw_gui();

            let (_output, paint_commands) = self.platform.end_frame(Some(&self.window));
            let paint_jobs = self.platform.context().tessellate(paint_commands);

            let mut egui_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("egui encoder"),
                    });

            let screen_descriptor = ScreenDescriptor {
                physical_width: self.surface_config.width,
                physical_height: self.surface_config.height,
                scale_factor: self.window.scale_factor() as f32,
            };
            self.egui_rpass.update_texture(
                &self.device,
                &self.queue,
                &self.platform.context().font_image(),
            );
            self.egui_rpass
                .update_user_textures(&self.device, &self.queue);
            self.egui_rpass.update_buffers(
                &self.device,
                &self.queue,
                &paint_jobs,
                &screen_descriptor,
            );
            self.egui_rpass
                .execute(
                    &mut egui_encoder,
                    &view,
                    &paint_jobs,
                    &screen_descriptor,
                    None,
                )
                .unwrap();

            self.queue.submit(std::iter::once(egui_encoder.finish()));
        }
        output.present();

        Ok(())
    }

    fn map_scene_item_to_model<'c>(models: &'c mut Vec<Model>, item: &SceneItem) -> &'c mut Model {
        models
            .iter_mut()
            .filter(|model| model.id == Uuid::parse_str(&*item.id).unwrap())
            .next()
            .unwrap()
    }

    fn draw_gui(&mut self) {
        egui::Window::new("Settings").show(&self.platform.context(), |ui| {
            ui.label("Clear color");
            ui.color_edit_button_rgb(&mut self.clear_color);
            ui.checkbox(&mut self.only_show_emissive, "Only show emissive materials");
            if ui.button("Reset Camera").clicked() {
                self.camera.eye = [0.0f32, 1.0f32, 2.0f32].into();
            }
            if ui.button("Reload world").clicked() {
                let (world, models) = State::load_world(
                    &self.device,
                    &self.queue,
                    &self.texture_bind_group_layout,
                    &self.simple_uniform_layout,
                    &self.simple_uniform_layout,
                );
                self.models = models;
                self.world = world;
            }
            if ui.button("Save world").clicked() {
                for m in self.models.as_slice() {
                    for i in self.world.scene.as_mut_slice() {
                        if Uuid::parse_str(&i.id).unwrap() == m.id {
                            println!("Saved model #{:?}", m.id);
                            i.position = m.model_offset;
                            i.rotation = m.model_rot;
                        }
                    }
                }
                let res_dir = env::current_dir().unwrap().join("renderer/res");
                fs::write(
                    res_dir.join("world.toml"),
                    toml::to_string(&self.world).expect("Couldn't serialize world"),
                )
                .expect("Couldn't write to file")
            }

            ui.separator();
            ui.add(
                egui::Slider::new(&mut self.green_duration, 1u32..=20u32)
                    .clamp_to_range(true)
                    .text("Green duration"),
            );
            ui.add(
                egui::Slider::new(&mut self.yellow_duration, 1u32..=20u32)
                    .clamp_to_range(true)
                    .text("Yellow duration"),
            );
            ui.add(
                egui::Slider::new(&mut self.red_yellow_duration, 1u32..=20u32)
                    .clamp_to_range(true)
                    .text("Red-Yellow duration"),
            );
            ui.add(
                egui::Slider::new(&mut self.all_red_duration, 1u32..=20u32)
                    .clamp_to_range(true)
                    .text("All-red duration"),
            );
            ui.label("Next cycle:");
            ui.label(format!("{}", self.next_cycle.as_secs()));
            if ui.button("Start / Stop").clicked() {
                if self.next_cycle > std::time::Duration::new(0, 0) {
                    self.next_cycle = std::time::Duration::new(0, 0);
                } else {
                    self.next_cycle = std::time::SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap();
                }
            }
        });
        egui::Window::new("Scene").show(&self.platform.context(), |ui| {
            for scene_item in self.world.scene.as_slice() {
                egui::CollapsingHeader::new(&scene_item.name).show(ui, |ui| {
                    let model_ref = State::map_scene_item_to_model(&mut self.models, scene_item);
                    egui::CollapsingHeader::new("Model").show(ui, |ui| {
                        ui.label("Position");
                        ui.add(
                            egui::Slider::new(&mut model_ref.model_offset[0], -20.0..=20.0)
                                .clamp_to_range(false)
                                .text("x"),
                        );
                        ui.add(
                            egui::Slider::new(&mut model_ref.model_offset[1], -20.0..=20.0)
                                .clamp_to_range(false)
                                .text("y"),
                        );
                        ui.add(
                            egui::Slider::new(&mut model_ref.model_offset[2], -20.0..=20.0)
                                .clamp_to_range(false)
                                .text("z"),
                        );
                        ui.label("Rotation");
                        ui.add(
                            egui::Slider::new(&mut model_ref.model_rot[0], 0.0..=360.0)
                                .clamp_to_range(false)
                                .text("x"),
                        );
                        ui.add(
                            egui::Slider::new(&mut model_ref.model_rot[1], 0.0..=360.0)
                                .clamp_to_range(false)
                                .text("y"),
                        );
                        ui.add(
                            egui::Slider::new(&mut model_ref.model_rot[2], 0.0..=360.0)
                                .clamp_to_range(false)
                                .text("z"),
                        );
                    });
                    for mesh in model_ref.meshes.as_mut_slice() {
                        egui::CollapsingHeader::new(format!("Mesh {}", mesh.name)).show(ui, |ui| {
                            ui.label("Position");
                            ui.add(
                                egui::Slider::new(&mut mesh.mesh_uniform.position[0], -20.0..=20.0)
                                    .clamp_to_range(false)
                                    .text("x"),
                            );
                            ui.add(
                                egui::Slider::new(&mut mesh.mesh_uniform.position[1], -20.0..=20.0)
                                    .clamp_to_range(false)
                                    .text("y"),
                            );
                            ui.add(
                                egui::Slider::new(&mut mesh.mesh_uniform.position[2], -20.0..=20.0)
                                    .clamp_to_range(false)
                                    .text("z"),
                            );
                            match model_ref.materials.as_mut_slice()[mesh.material as usize] {
                                MaterialStructs::EMISSION(ref mut mat) => {
                                    egui::CollapsingHeader::new("Emission").show(ui, |ui| {
                                        ui.add(
                                            egui::Slider::new(&mut mat.uniform.constant, 0.0..=2.0)
                                                .text("Constant"),
                                        );
                                        ui.add(
                                            egui::Slider::new(&mut mat.uniform.linear, 0.0..=1.0)
                                                .text("Linear"),
                                        );
                                        ui.add(
                                            egui::Slider::new(
                                                &mut mat.uniform.quadratic,
                                                0.0..=1.0,
                                            )
                                            .text("Quadratic"),
                                        );
                                        ui.label("Diffuse");
                                        ui.color_edit_button_rgb(&mut mat.uniform.diffuse);
                                        ui.label("Ambient");
                                        ui.color_edit_button_rgb(&mut mat.uniform.ambient);
                                        ui.label("Specular");
                                        ui.color_edit_button_rgb(&mut mat.uniform.specular);
                                    });
                                }
                                _ => {}
                            };
                        });
                    }
                });
            }
        });
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(Size::from(LogicalSize {
            width: 1400,
            height: 800,
        }))
        .build(&event_loop)
        .expect("Can't open window");

    let mut state = pollster::block_on(State::new(window));
    event_loop.run(move |event, _, control_flow| {
        state.platform.handle_event(&event);
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            debug!(
                                "Resized window! New size: {} {}",
                                physical_size.width, physical_size.height
                            );
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window.id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window.request_redraw();
            }
            _ => {}
        }
    });
}
