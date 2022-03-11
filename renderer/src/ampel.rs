use crate::model::EmissiveMaterial;
use crate::{MaterialStructs, Model};

pub enum AmpelStatus {
    RED,
    REDYELLOW,
    GREEN,
    YELLOW,
}

pub struct Ampel {}

impl Ampel {
    pub fn set_ampel_status(model: &mut Model, status: AmpelStatus) {
        let mut mat_red: Option<&mut EmissiveMaterial> = None;
        let mut mat_yellow: Option<&mut EmissiveMaterial> = None;
        let mut mat_green: Option<&mut EmissiveMaterial> = None;

        for m_ in model.materials.as_mut_slice() {
            match m_ {
                MaterialStructs::EMISSION(mat) => match mat.name.as_str() {
                    "Ampel.Rot" => mat_red = Some(mat),
                    "Ampel.Gelb" => mat_yellow = Some(mat),
                    "Ampel.Gruen" => mat_green = Some(mat),
                    _ => {}
                },
                _ => continue,
            }
        }

        match status {
            AmpelStatus::RED => {
                mat_red.as_mut().map(|mut s| {
                    s.uniform.quadratic = 0.032;
                    s
                });
                mat_red.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.8, 0.0, 0.011073];
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.033, 0.031, 0.004];
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.002, 0.027, 0.002];
                    s
                });
            }
            AmpelStatus::REDYELLOW => {
                mat_red.as_mut().map(|mut s| {
                    s.uniform.quadratic = 0.032;
                    s
                });
                mat_red.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.8, 0.0, 0.011073];
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.quadratic = 0.032;
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.8, 0.547093, 0.0];
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.002, 0.027, 0.002];
                    s
                });
            }
            AmpelStatus::YELLOW => {
                mat_red.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_red.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.027, 0.002, 0.001];
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.quadratic = 0.032;
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.8, 0.547093, 0.0];
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.002, 0.027, 0.002];
                    s
                });
            }
            AmpelStatus::GREEN => {
                mat_red.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_red.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.027, 0.002, 0.001];
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.033, 0.031, 0.004];
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.quadratic = 0.032;
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.016485, 0.800000, 0.0];
                    s
                });
            }
        }
    }
}
