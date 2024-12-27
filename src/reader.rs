use smolmatrix::*;
use std::io;

static TO_OUR: [u8; 47] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    1,
    18,
    19,
    1,
    20,
    21,
    0,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    9,
    41,
    42,
];

pub fn read_images<R: io::Read>(b: &mut R, mut limit: usize) -> io::Result<Vec<Vector<784>>> {
    b.read_exact(&mut [0; 4])?;

    let mut size = [0; 4];
    b.read_exact(&mut size)?;

    let mut images = Vec::with_capacity(u32::from_be_bytes(size) as usize);

    b.read_exact(&mut [0; 8])?;

    let mut buf = [0; 784];
    while let (Ok(()), true) = (b.read_exact(&mut buf), limit > 0) {
        let mut v = Vector::new_zeroed();

        for (yi, y) in buf.chunks(28).enumerate() {
            for (xi, i) in y.iter().enumerate() {
                v[yi + xi * 28] = *i as f32 / 255.0;
            }
        }

        images.push(v);
        limit -= 1;
    }

    Ok(images)
}

pub fn read_labels<R: io::Read>(b: &mut R, mut limit: usize) -> io::Result<Vec<u8>> {
    b.read_exact(&mut [0; 4])?;

    let mut size = [0; 4];
    b.read_exact(&mut size)?;

    let mut labels = Vec::with_capacity(u32::from_be_bytes(size) as usize);
    let mut buf = [0; 1];

    while let (Ok(()), true) = (b.read_exact(&mut buf), limit > 0) {
        labels.push(TO_OUR[buf[0] as usize]);
        limit -= 1;
    }

    Ok(labels)
}

pub fn read_data(t: &str, limit: Option<usize>) -> io::Result<(Vec<Vector<784>>, Vec<u8>)> {
    let limit = limit.unwrap_or(usize::MAX);
    let mut images = std::fs::File::open(format!("db/emnist-balanced-{t}-images-idx3-ubyte"))?;
    let images = read_images(&mut images, limit)?;
    let mut labels = std::fs::File::open(format!("db/emnist-balanced-{t}-labels-idx1-ubyte"))?;
    let labels = read_labels(&mut labels, limit)?;

    Ok((images, labels))
}
