use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct SceneItem {
    pub model_file: String,
    pub name: String,
    pub id: String,
    pub rotation: [f32; 4],
    pub position: [f32; 3],
}

#[derive(Deserialize, Serialize)]
pub(crate) struct World {
    pub name: String,
    pub scene: Vec<SceneItem>,
}
