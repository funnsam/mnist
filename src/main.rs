use smolmatrix::*;
use smolnn2::*;
use mnist::MnistModel;

mod reader;

const MINI_BATCH_SIZE: usize = 5000;

fn load<T: for<'a> serde::Deserialize<'a>, F: Fn() -> T>(n: usize, default: F) -> T {
    std::env::args().nth(n)
        .and_then(|path| std::fs::read(path).ok())
        .and_then(|file| postcard::from_bytes(&file).ok())
        .unwrap_or_else(default)
}

fn preproc(i: &Vector<784>) -> Vector<784> {
    let mut result = Vector::new_zeroed();

    let off_x = fastrand::isize(-3..=3);
    let off_y = fastrand::isize(-3..=3);

    let theta = fastrand::f32() * 0.5 - 0.25;
    let r_mat = matrix!(
        2 x 2
        [theta.cos(), theta.sin()]
        [-theta.sin(), theta.cos()]
    );

    for yi in 0..28 {
        for xi in 0..28 {
            let x = xi as isize + off_x - 14;
            let y = yi as isize + off_y - 14;
            let v = &r_mat * &vector!(2 [x as f32, y as f32]);
            let x = v.x().round() as isize + 14;
            let y = v.y().round() as isize + 14;

            if 0 <= x && x < 28 && 0 <= y && y < 28 {
                result[(x + y * 28) as usize] = i[xi + yi * 28];
            }
        }
    }

    result + &Vector::from_iter(core::iter::repeat_with(|| {
        let x = fastrand::f32();
        let sigma = 0.0005;
        (-(x * x) / (2.0 * sigma)).exp() / (6400.0 * sigma * core::f32::consts::TAU.sqrt())
    }))
}

fn main() {
    let mut model = load(2, || MnistModel {
        l1: fcnn::Fcnn::new_he_uniform(fastrand::f32).into(),
        l2: activation::LeakyRelu(0.01),
        l3: fcnn::Fcnn::new_he_uniform(fastrand::f32).into(),
        l4: activation::LeakyRelu(0.01),
        l5: fcnn::Fcnn::new_xavier_uniform(fastrand::f32).into(),
        l6: activation::Softmax,
    });

    match std::env::args().nth(1).unwrap().as_str() {
        "train" => {
            let (images, labels) = reader::read_data("train", None).unwrap();

            let (mut o1, mut o3, mut o5) = load(3, || (
                Box::new(fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.005))),
                Box::new(fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.005))),
                Box::new(fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.005))),
            ));

            for i in 0.. {
                let mut c1 = fcnn::FcnnCollector::new();
                let mut c3 = fcnn::FcnnCollector::new();
                let mut c5 = fcnn::FcnnCollector::new();
                let mut loss = 0.0;

                for _ in 0..MINI_BATCH_SIZE {
                    let i = fastrand::usize(0..images.len());
                    let input = &preproc(&images[i]);
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
            let (images, labels) = reader::read_data("t10k", None).unwrap();

            let mut wrong_c = [0; 10];

            for (i, l) in images.iter().zip(labels.iter()) {
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

                if p.0 != *l as usize {
                    wrong_c[*l as usize] += 1;
                }
            }

            println!("Correct%: {:.2}%", (images.len() - wrong_c.iter().sum::<usize>()) as f32 / images.len() as f32 * 100.0);
            println!("Incorrect frequency: {wrong_c:?}");
        },
        "visualize_train" => {
            let (images, labels) = reader::read_data("train", None).unwrap();

            for (i, l) in images.iter().zip(labels.iter()) {
                visualize(&preproc(i));
                println!("{l}");
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        },
        _ => panic!("unknown command"),
    }
}

#[allow(unused)]
fn visualize(fb: &Vector<784>) {
    for yhalf in 0..28 / 2 {
        for x in 0..28 {
            plot(fb[(0, x + yhalf * 56)], fb[(0, x + yhalf * 56 + 28)]);
        }

        println!("\x1b[0m");
    }
}

#[allow(unused)]
fn plot(a: f32, b: f32) {
    let a = (a * 255.0) as u8;
    let b = (b * 255.0) as u8;
    print!("\x1b[38;2;{a};{a};{a}m\x1b[48;2;{b};{b};{b}m▀");
}

#[allow(unused)]
fn bar(v: &Vector<10>) {
    const BAR_LENGTH: usize = 30;

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
