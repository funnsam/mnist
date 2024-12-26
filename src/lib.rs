use smolnn2::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

model! {
    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub MnistModel: 1, 784 => 1, 10

    => pub Box<fcnn::Fcnn<784, 128>>
    => #[activation] pub activation::LeakyRelu
    => pub Box<fcnn::Fcnn<128, 64>>
    => #[activation] pub activation::LeakyRelu
    => pub Box<fcnn::Fcnn<64, 10>>
    => #[activation] pub activation::Softmax
}

pub fn load<T: for<'a> serde::Deserialize<'a>, F: Fn() -> T>(b: &[u8], default: F) -> T {
    postcard::from_bytes(b).ok()
        .unwrap_or_else(default)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Model(MnistModel);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn load_model(b: &[u8]) -> Model {
    Model(load::<MnistModel, _>(b, || panic!()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_prob(m: &Model, c: &[u8]) -> Vec<f32> {
    let i = smolmatrix::Vector::from_iter(c.chunks(4).map(|i| i[0] as f32 / 255.0));
    m.0.forward(&i).inner.into_iter().flatten().collect()
}
