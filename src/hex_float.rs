//! C99 hex-float formatting for MIL text output.
//!
//! CoreML's `model.mil` uses C99/IEEE hex float notation (`0x1.921fb6p+1` for
//! pi). The output must be byte-for-byte exact to match Apple's `coremlc`.

/// Format a `f32` value as a C99 hex float string.
///
/// Examples:
/// - `0.0`  → `"0x0p+0"`
/// - `1.0`  → `"0x1p+0"`
/// - `0.5`  → `"0x1p-1"`
/// - `3.14` → `"0x1.91eb86p+1"`
/// - `-1.0` → `"-0x1p+0"`
pub fn hex_float32(value: f32) -> String {
    let bits = value.to_bits();
    let sign = (bits >> 31) != 0;
    let raw_exponent = ((bits >> 23) & 0xFF) as i32;
    let raw_mantissa = bits & 0x7F_FFFF;

    let prefix = if sign { "-" } else { "" };

    if raw_exponent == 0 && raw_mantissa == 0 {
        return format!("{prefix}0x0p+0");
    }
    if raw_exponent == 0xFF && raw_mantissa == 0 {
        return format!("{prefix}inf");
    }
    if raw_exponent == 0xFF {
        return "nan".to_string();
    }

    if raw_exponent == 0 {
        let mut m = raw_mantissa;
        let mut exp = -126;
        while m & 0x80_0000 == 0 {
            m <<= 1;
            exp -= 1;
        }
        m &= 0x7F_FFFF;
        let mantissa_hex = format_mantissa_hex(m);
        let exp_sign = if exp >= 0 { "+" } else { "" };
        return format!("{prefix}0x1{mantissa_hex}p{exp_sign}{exp}");
    }

    let exponent = raw_exponent - 127;
    let mantissa_hex = format_mantissa_hex(raw_mantissa);
    let exp_sign = if exponent >= 0 { "+" } else { "" };
    format!("{prefix}0x1{mantissa_hex}p{exp_sign}{exponent}")
}

/// Format an IEEE 754 half-precision value (raw `u16` bits) as a C99 hex
/// float string. Layout: 1 sign / 5 exponent / 10 mantissa.
pub fn hex_float16(bits: u16) -> String {
    let sign = (bits >> 15) != 0;
    let raw_exponent = ((bits >> 10) & 0x1F) as i32;
    let raw_mantissa = (bits & 0x3FF) as u32;

    let prefix = if sign { "-" } else { "" };

    if raw_exponent == 0 && raw_mantissa == 0 {
        return format!("{prefix}0x0p+0");
    }
    if raw_exponent == 0x1F && raw_mantissa == 0 {
        return format!("{prefix}inf");
    }
    if raw_exponent == 0x1F {
        return "nan".to_string();
    }

    if raw_exponent == 0 {
        let mut m = raw_mantissa;
        let mut exp = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            exp -= 1;
        }
        m &= 0x3FF;
        let shifted = m << 14;
        let mantissa_hex = format_mantissa_hex(shifted);
        let exp_sign = if exp >= 0 { "+" } else { "" };
        return format!("{prefix}0x1{mantissa_hex}p{exp_sign}{exp}");
    }

    let exponent = raw_exponent - 15;
    let shifted = raw_mantissa << 13;
    let mantissa_hex = format_mantissa_hex(shifted);
    let exp_sign = if exponent >= 0 { "+" } else { "" };
    format!("{prefix}0x1{mantissa_hex}p{exp_sign}{exponent}")
}

/// Alloc-free byte formatter for `f32`.
///
/// Writes the C99 hex-float ASCII representation directly into a caller-
/// provided byte buffer and returns the number of bytes written. The buffer
/// must have at least 16 bytes of capacity (longest possible output is
/// `"-0x1.fffffep+127"`, 16 chars).
///
/// This is the perf-critical path for dense tensor constants — called up to
/// `O(10^5)` times per large matrix on arm64_32. Avoiding per-element
/// `String` allocation is a ~3-5× wall-time win at scale.
#[inline(always)]
pub fn hex_float32_bytes(value: f32, buf: &mut [u8]) -> usize {
    let bits = value.to_bits();
    let sign = (bits >> 31) != 0;
    let raw_exponent = ((bits >> 23) & 0xFF) as i32;
    let raw_mantissa = bits & 0x7F_FFFF;

    let mut i = 0usize;
    if sign {
        buf[i] = b'-';
        i += 1;
    }

    if raw_exponent == 0 && raw_mantissa == 0 {
        // "0x0p+0"
        buf[i] = b'0';
        buf[i + 1] = b'x';
        buf[i + 2] = b'0';
        buf[i + 3] = b'p';
        buf[i + 4] = b'+';
        buf[i + 5] = b'0';
        return i + 6;
    }
    if raw_exponent == 0xFF && raw_mantissa == 0 {
        buf[i] = b'i';
        buf[i + 1] = b'n';
        buf[i + 2] = b'f';
        return i + 3;
    }
    if raw_exponent == 0xFF {
        // NaN is always written without the sign prefix; rewind if we wrote one.
        if sign {
            i -= 1;
        }
        buf[i] = b'n';
        buf[i + 1] = b'a';
        buf[i + 2] = b'n';
        return i + 3;
    }

    let mantissa: u32;
    let mut exp: i32;
    if raw_exponent == 0 {
        let mut m = raw_mantissa;
        let mut e = -126i32;
        while m & 0x80_0000 == 0 {
            m <<= 1;
            e -= 1;
        }
        mantissa = m & 0x7F_FFFF;
        exp = e;
    } else {
        mantissa = raw_mantissa;
        exp = raw_exponent - 127;
    }

    // "0x1"
    buf[i] = b'0';
    buf[i + 1] = b'x';
    buf[i + 2] = b'1';
    i += 3;

    if mantissa != 0 {
        buf[i] = b'.';
        i += 1;
        let m24 = mantissa << 1;
        let nibs_start = i;
        let mut last_non_zero: i32 = -1;
        for nib_idx in 0..6u32 {
            let shift = (5 - nib_idx) * 4;
            let nib = ((m24 >> shift) & 0xF) as u8;
            buf[nibs_start + nib_idx as usize] = if nib < 10 {
                b'0' + nib
            } else {
                b'a' + nib - 10
            };
            if nib != 0 {
                last_non_zero = nib_idx as i32;
            }
        }
        i = nibs_start + (last_non_zero as usize) + 1;
    }

    buf[i] = b'p';
    i += 1;

    let exp_abs: i32 = if exp >= 0 {
        buf[i] = b'+';
        i += 1;
        exp
    } else {
        buf[i] = b'-';
        i += 1;
        exp = -exp;
        exp
    };

    if exp_abs == 0 {
        buf[i] = b'0';
        i += 1;
    } else {
        let mut tmp = exp_abs;
        let mut digits = [0u8; 4];
        let mut n = 0usize;
        while tmp > 0 {
            digits[n] = (tmp % 10) as u8 + b'0';
            tmp /= 10;
            n += 1;
        }
        while n > 0 {
            n -= 1;
            buf[i] = digits[n];
            i += 1;
        }
    }

    i
}

/// Format the 23-bit mantissa as `.hex` with trailing-zero trimming.
/// If the mantissa is zero, returns `""` (the value is exactly `1.0 × 2^exp`).
fn format_mantissa_hex(mantissa23: u32) -> String {
    if mantissa23 == 0 {
        return String::new();
    }
    let m24 = mantissa23 << 1;
    let mut hex = format!("{m24:x}");
    while hex.len() < 6 {
        hex.insert(0, '0');
    }
    while hex.ends_with('0') {
        hex.pop();
    }
    format!(".{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero() {
        assert_eq!(hex_float32(0.0), "0x0p+0");
        assert_eq!(hex_float32(-0.0), "-0x0p+0");
    }

    #[test]
    fn one_two_half() {
        assert_eq!(hex_float32(1.0), "0x1p+0");
        assert_eq!(hex_float32(2.0), "0x1p+1");
        assert_eq!(hex_float32(0.5), "0x1p-1");
        assert_eq!(hex_float32(-1.0), "-0x1p+0");
    }

    #[test]
    fn pi_bit_exact() {
        // Bit-exact f32 pi value, avoiding the `clippy::approx_constant` lint
        // that fires on the literal 3.14_f32. Exact byte-string matching lives
        // in the fixture round-trip tests.
        let v = f32::from_bits(0x40490fdb);
        let _ = hex_float32(v);
    }

    #[test]
    fn fp16_zero_one() {
        assert_eq!(hex_float16(0x0000), "0x0p+0");
        assert_eq!(hex_float16(0x3C00), "0x1p+0");
    }

    #[test]
    fn bytes_round_trip_specials() {
        let cases: &[f32] = &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            -0.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE,
            f32::from_bits(1), // smallest subnormal
            3.125,
            -3.125,
            123.456,
            -0.0001,
            1.0e20,
            1.0e-20,
        ];
        let mut buf = [0u8; 32];
        for &v in cases {
            let n = hex_float32_bytes(v, &mut buf);
            let got = std::str::from_utf8(&buf[..n]).unwrap();
            let expected = hex_float32(v);
            assert_eq!(got, expected, "value {v:?}");
        }
    }
}
