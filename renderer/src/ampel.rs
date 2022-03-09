use crate::{MaterialStructs, Model};
use crate::model::{EmissiveMaterial};

pub enum AmpelStatus {
    RED,
    REDYELLOW,
    GREEN,
    YELLOW,
}

pub(crate) struct Ampel<'a> {
    mat_red: Option<&'a mut EmissiveMaterial>,
    mat_yellow: Option<&'a mut EmissiveMaterial>,
    mat_green: Option<&'a mut EmissiveMaterial>,
}

impl<'a> Ampel<'a> {
    pub(crate) fn from_model(model: &'a mut Model) -> Option<Ampel<'a>> {
        let mut mat_red: Option<&mut EmissiveMaterial> = None;
        let mut mat_yellow: Option<&mut EmissiveMaterial> = None;
        let mut mat_green: Option<&mut EmissiveMaterial> = None;
        for m_ in model.materials.as_mut_slice() {
            match m_ {
                MaterialStructs::EMISSION(mat) => {
                    match mat.name.as_str() {
                        "Ampel.Rot" => {mat_red = Some(mat)}
                        "Ampel.Gelb" => {mat_yellow = Some(mat)}
                        "Ampel.Gruen" => {mat_green = Some(mat)}
                        _ => {}
                    }
                }
                _ => {continue}
            }
        }
        Some(Self {
            mat_red,
            mat_yellow,
            mat_green,
        })
    }

    pub fn set_status(&mut self, status: AmpelStatus) {
        match status {
            AmpelStatus::RED => {
                self.mat_red.as_mut().map(|mut s| {s.uniform.quadratic = 0.032; s});
                self.mat_red.as_mut().map(|mut s| {s.uniform.diffuse = [0.8, 0.0, 0.011073]; s});
                self.mat_yellow.as_mut().map(|mut s| {s.uniform.quadratic = 100.0; s});
                self.mat_yellow.as_mut().map(|mut s| {s.uniform.diffuse = [0.033, 0.031, 0.004]; s});
                self.mat_green.as_mut().map(|mut s| {s.uniform.quadratic = 100.0; s});
                self.mat_green.as_mut().map(|mut s| {s.uniform.diffuse = [0.002, 0.027, 0.002]; s});
            }
            AmpelStatus::REDYELLOW => {
                self.mat_red.as_mut().map(|mut s| {s.uniform.quadratic = 0.032; s});
                self.mat_red.as_mut().map(|mut s| {s.uniform.diffuse = [0.8, 0.0, 0.011073]; s});
                self.mat_yellow.as_mut().map(|mut s| {s.uniform.quadratic = 0.032; s});
                self.mat_yellow.as_mut().map(|mut s| {s.uniform.diffuse = [0.8, 0.547093, 0.0]; s});
                self.mat_green.as_mut().map(|mut s| {s.uniform.quadratic = 100.0; s});
                self.mat_green.as_mut().map(|mut s| {s.uniform.diffuse = [0.002, 0.027, 0.002]; s});
            }
            AmpelStatus::YELLOW => {
                self.mat_red.as_mut().map(|mut s| {s.uniform.quadratic = 100.0; s});
                self.mat_red.as_mut().map(|mut s| {s.uniform.diffuse = [0.027, 0.002, 0.001]; s});
                self.mat_yellow.as_mut().map(|mut s| {s.uniform.quadratic = 0.032; s});
                self.mat_yellow.as_mut().map(|mut s| {s.uniform.diffuse = [0.8, 0.547093, 0.0]; s});
                self.mat_green.as_mut().map(|mut s| {s.uniform.quadratic = 100.0; s});
                self.mat_green.as_mut().map(|mut s| {s.uniform.diffuse = [0.002, 0.027, 0.002]; s});
            }
            AmpelStatus::GREEN => {
                self.mat_red.as_mut().map(|mut s| {s.uniform.quadratic = 100.0; s});
                self.mat_red.as_mut().map(|mut s| {s.uniform.diffuse = [0.027, 0.002, 0.001]; s});
                self.mat_yellow.as_mut().map(|mut s| {s.uniform.quadratic = 100.0; s});
                self.mat_yellow.as_mut().map(|mut s| {s.uniform.diffuse = [0.033, 0.031, 0.004]; s});
                self.mat_green.as_mut().map(|mut s| {s.uniform.quadratic = 0.032; s});
                self.mat_green.as_mut().map(|mut s| {s.uniform.diffuse = [0.016485, 0.800000, 0.0]; s});
            }
        }
    }
}