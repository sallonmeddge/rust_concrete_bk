extern crate concrete;
use concrete::*;

fn main() -> Result<(), CryptoAPIError> {

    // generate a secret key
    let secret_key = LWESecretKey::new(&LWE128_630);

    // the two values to add
    let m1 = 8.2;
    let m2 = 5.6;

    // Encode in [0, 10[ with 8 bits of precision and 1 bit of padding
    let encoder = Encoder::new(0., 10., 8, 1)?;

    // encrypt plaintexts
    let mut c1 = LWE::encode_encrypt(&secret_key, m1, &encoder)?;
    let c2 = LWE::encode_encrypt(&secret_key, m2, &encoder)?;

    // add the two ciphertexts homomorphically, and store in c1
    c1.add_with_padding_inplace(&c2)?;

    // decrypt and decode the result
    let m3 = c1.decrypt_decode(&secret_key)?;

    // print the result and compare to non-FHE addition
    println!("Real: {}, FHE: {}", m1 + m2, m3);

    Ok(())
}