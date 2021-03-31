// extern crate byteorder;
extern crate concrete;
use concrete::*;
use std::io::prelude::*;
use std::fs::File;
use std::{io, str, fmt};
use std::io::Read;
use std::str::FromStr;
use std::path::Path;
use std::f32;
// use byteorder::{BigEndian, WriteBytesExt};

fn main() -> std::io::Result<()> {
    // let data = get_txt("data/train_x_60000x784_float32.txt");
    let mut buffer = File::create("foo_test.txt")?;

    let secret_key = LWESecretKey::new(&LWE128_630);
    let encoder = Encoder::new(0., 255., 8, 1).unwrap();

    if let Ok(lines) = read_lines("data/train_x_60000x784_float32.txt") {
        // Consumes the iterator, returns an (Optional) String
        let mut i = 0;
        for line in lines {
            if i > 3 {
                break
            }
            if let Ok(ip) = line {
                let temp: Vec<&str> = ip.split(' ').collect();
                // print_type_of(&temp);
                let mut j  = 0;
                for t in temp {
                    // print_type_of(&t);
                    let num_k: i32 = t.to_string().parse().unwrap();
                    print!("beginzzzz {:?}, {:?} end \n ",num_k, j);

                    let f64_k: f64 = num_k as f64;
                    let mut c1 = LWE::encode_encrypt(&secret_key, f64_k, &encoder).unwrap();
                    
                    let z = c1.ciphertext.get_body();
                    print!("begin {:?}, {:?}, {:?}, {:?} end \n ",num_k, c1, z, j);
                    j += 1;
                    // for i in &data{                                                                                                                                                                  
                    //     buffer.write_all((*i)).expect("Unable to write data");                                                                                                                            
                    // }  
                    // let fzzz :&[u8];
                    // if j < 784 {
                    //     let sp = " ";
                    //     z.push_str(sp);
                    // } else{
                    //     // fzzz = z.as_bytes();
                    // }
                    // fzzz = z.as_bytes();
                    // j += 1;
                    // let bytes_written = buffer.write(&fzzz)?;
                }

            }
            let new_line = "\n".as_bytes();

            let bytes_written = buffer.write(&new_line)?;
            i += 1;
        }
    }

    // let mut pos = 0;
    // let mut buffer = File::create("foo.txt")?;

    // let lines: Vec<Vec<&str>> = data.trim().split('\n').map(|line| line.split(' ').collect()).collect();
    // let rows = lines.len();
    // let columns = lines[0].len();
    // print!("{} {}\n",rows, columns);
    // print!("{}\n",lines[rows-1][columns-1]);
    // print_type_of(&lines[rows-1][columns-1]);
    // print!("{}",lines[rows-1][columns-1].len());
    // while pos < data.len() {
    //     let bytes_written = buffer.write(&data[pos..])?;
    //     pos += bytes_written;
    // }
    Ok(())
}
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

// fn write_floats(v: &[f32], f: &mut std:fs::File) -> std::io::Result<()> {
//     for float in v {
//         f.write_f64<LittleEndian>(float)?;
//     }
//     Ok(())
// }