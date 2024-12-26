use smolmatrix::*;
use smolnn2::*;

mod reader;

const BAR_LENGTH: usize = 30;
const MINI_BATCH_SIZE: usize = 10000;

model! {
    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub MnistModel: 1, 784 => 1, 10

    => fcnn::Fcnn<784, 64>
    => #[activation] activation::Tanh
    => fcnn::Fcnn<64, 64>
    => #[activation] activation::Tanh
    => fcnn::Fcnn<64, 10>
    => #[activation] activation::Softmax
}

fn load<T: for<'a> serde::Deserialize<'a>, F: Fn() -> T>(n: usize, default: F) -> T {
    std::env::args().nth(n)
        .and_then(|path| std::fs::read(path).ok())
        .and_then(|file| postcard::from_bytes(&file).ok())
        .unwrap_or_else(default)
}

fn main() {
    let (images, labels) = reader::read_data("train", None).unwrap();
    let mut model = load(2, || MnistModel {
        l1: fcnn::Fcnn::new_xavier_uniform(fastrand::f32),
        l2: activation::Tanh,
        l3: fcnn::Fcnn::new_xavier_uniform(fastrand::f32),
        l4: activation::Tanh,
        l5: fcnn::Fcnn::new_xavier_uniform(fastrand::f32),
        l6: activation::Softmax,
    });

    match std::env::args().nth(1).unwrap().as_str() {
        "train" => {
            let (mut o1, mut o3, mut o5) = load(3, || (
                fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.001)),
                fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.001)),
                fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.001)),
            ));

            for i in 0.. {
                let mut c1 = fcnn::FcnnCollector::new();
                let mut c3 = fcnn::FcnnCollector::new();
                let mut c5 = fcnn::FcnnCollector::new();
                let mut loss = 0.0;

                for _ in 0..MINI_BATCH_SIZE {
                    let i = fastrand::usize(0..images.len());
                    let input = &images[i];
                    let label = labels[i];
                    let mut expected = Vector::new_zeroed();
                    expected[label as usize] = 1.0;

                    let out = model.back_propagate(
                        input,
                        &expected,
                        loss::categorical_cross_entropy_derivative,
                        &mut c1,
                        &mut c3,
                        &mut c5,
                    );
                    loss += loss::categorical_cross_entropy(out, &expected).inner.iter().flatten().sum::<f32>();
                }

                model.l1.update(c1, MINI_BATCH_SIZE, &mut o1);
                model.l3.update(c3, MINI_BATCH_SIZE, &mut o3);
                model.l5.update(c5, MINI_BATCH_SIZE, &mut o5);

                println!("{i:>5} {}", loss / MINI_BATCH_SIZE as f32);

                if i % 10 == 0 {
                    std::fs::write("model.bin", postcard::to_stdvec(&model).unwrap()).unwrap();
                    std::fs::write("adams.bin", postcard::to_stdvec(&(&o1, &o3, &o5)).unwrap()).unwrap();
                }
            }
        },
        "try" => {
            for (i, l) in images.iter().zip(labels.iter()) {
                visualize(i);
                println!("Expected: {}", l);

                let f = model.forward(i);
                let p = f
                    .inner
                    .iter()
                    .enumerate()
                    .max_by(|a, b| {
                        a.1[0]
                            .partial_cmp(&b.1[0])
                            .unwrap_or(core::cmp::Ordering::Equal)
                    })
                .unwrap();

                println!("Predicted: {} ({:.1}%)", p.0, p.1[0] * 100.0);
                bar(&f);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        },
        _ => panic!("unknown command"),
    }
}

fn visualize(fb: &Vector<784>) {
    for yhalf in 0..28 / 2 {
        for x in 0..28 {
            plot(fb[(0, x + yhalf * 56)], fb[(0, x + yhalf * 56 + 28)]);
        }

        println!("\x1b[0m");
    }
}

fn plot(a: f32, b: f32) {
    let a = (a * 255.0) as u8;
    let b = (b * 255.0) as u8;
    print!("\x1b[38;2;{a};{a};{a}m\x1b[48;2;{b};{b};{b}m▀");
}

fn bar(v: &Vector<10>) {
    for (i, [v]) in v.inner.iter().enumerate() {
        let len = ((v + 1.0).log2() * BAR_LENGTH as f32).floor() as usize;
        println!(
            "{i} {:━<len$}{:<2$} ({3:.1}%)",
            "",
            "",
            BAR_LENGTH - len,
            v * 100.0
        );
    }
}
