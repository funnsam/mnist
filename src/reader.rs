use smolmatrix::*;
use std::io;

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
                v[xi + yi * 28] = *i as f32 / 255.0;
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
        labels.push(buf[0]);
        limit -= 1;
    }

    Ok(labels)
}

pub fn read_data(t: &str, limit: Option<usize>) -> io::Result<(Vec<Vector<784>>, Vec<u8>)> {
    let limit = limit.unwrap_or(usize::MAX);
    let mut images = std::fs::File::open(format!("db/{t}-images-idx3-ubyte"))?;
    let images = read_images(&mut images, limit)?;
    let mut labels = std::fs::File::open(format!("db/{t}-labels-idx1-ubyte"))?;
    let labels = read_labels(&mut labels, limit)?;

    Ok((images, labels))
}
