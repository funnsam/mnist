use smolnn2::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

model! {
    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub MnistModel: 1, 784 => 1, 43

    => pub Box<fcnn::Fcnn<784, 386>>
    => #[activation] pub activation::LeakyRelu
    => pub Box<fcnn::Fcnn<386, 256>>
    => #[activation] pub activation::LeakyRelu
    => pub Box<fcnn::Fcnn<256, 43>>
    => #[activation] pub activation::Softmax
}

pub fn load<T: for<'a> serde::Deserialize<'a>, F: Fn() -> T>(b: &[u8], default: F) -> T {
    postcard::from_bytes(b).ok()
        .unwrap_or_else(default)
}

pub static MAP: [&str; 43] = [
    "0/O/o",
    "1/I/i/L/l",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9/q",
    "A",
    "B",
    "C/c",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J/j",
    "K/k",
    "M/m",
    "N",
    "P/p",
    "Q",
    "R",
    "S/s",
    "T",
    "U/u",
    "V/v",
    "W/w",
    "X/x",
    "Y/y",
    "Z/z",
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "n",
    "r",
    "t",
];

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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn map(i: usize) -> String { MAP[i].to_string() }
