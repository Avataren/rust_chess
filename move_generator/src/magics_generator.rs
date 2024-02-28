use crate::masks::{BISHOP_MASKS, ROOK_MASKS};
use core::panic;
use rand::Rng;

pub struct MagicGenerator {}

impl MagicGenerator {
    /// Transforms a 64-bit block using a magic number and the number of bits to use.
    fn transform(b: u64, magic: u64, bits: i32) -> i32 {
        (((b.wrapping_mul(magic)) >> (64 - bits)) & ((1 << bits) - 1) as u64) as i32
    }

    /// Counts the number of set bits in a 64-bit unsigned integer.
    // fn count_1s(mut b: u64) -> i32 {
    //     let mut r = 0;
    //     while b != 0 {
    //         r += 1;
    //         b &= b - 1;
    //     }
    //     r
    // }

    fn index_to_uint64(index: i32, bits: i32, mut mask: u64) -> u64 {
        let mut result: u64 = 0;
        let mut j: i32 = 0;
        for i in 0..bits {
            let mut bit = mask & !(mask - 1); // Isolate the lowest bit of the mask
            mask &= mask - 1; // Clear the lowest bit of the mask
            if (index & (1 << i)) != 0 {
                result |= bit;
            }
            while bit != 0 {
                bit >>= 1;
                j += 1;
            }
        }
        result
    }

    fn rmask(sq: i32) -> u64 {
        let mut result: u64 = 0;
        let rk = sq / 8;
        let fl = sq % 8;
        for r in (rk + 1)..7 {
            result |= 1 << (fl + r * 8);
        }
        for r in (1..rk).rev() {
            result |= 1 << (fl + r * 8);
        }
        for f in (fl + 1)..7 {
            result |= 1 << (f + rk * 8);
        }
        for f in (1..fl).rev() {
            result |= 1 << (f + rk * 8);
        }
        result
    }

    fn bmask(sq: i32) -> u64 {
        let mut result: u64 = 0;
        let rk = sq / 8;
        let fl = sq % 8;
        let mut r;
        let mut f;
        r = rk + 1;
        f = fl + 1;
        while r <= 6 && f <= 6 {
            result |= 1 << (f + r * 8);
            r += 1;
            f += 1;
        }
        r = rk + 1;
        f = fl - 1;
        while r <= 6 && f >= 1 {
            result |= 1 << (f + r * 8);
            r += 1;
            f -= 1;
        }
        r = rk - 1;
        f = fl + 1;
        while r >= 1 && f <= 6 {
            result |= 1 << (f + r * 8);
            r -= 1;
            f += 1;
        }
        r = rk - 1;
        f = fl - 1;
        while r >= 1 && f >= 1 {
            result |= 1 << (f + r * 8);
            r -= 1;
            f -= 1;
        }
        result
    }

    fn ratt(sq: i32, block: u64) -> u64 {
        let mut result: u64 = 0;
        let rk = sq / 8; // Rank
        let fl = sq % 8; // File

        // Positive rank direction
        for r in (rk + 1)..8 {
            result |= 1 << (fl + r * 8);
            if block & (1 << (fl + r * 8)) != 0 {
                break;
            }
        }

        // Negative rank direction
        for r in (0..rk).rev() {
            result |= 1 << (fl + r * 8);
            if block & (1 << (fl + r * 8)) != 0 {
                break;
            }
        }

        // Positive file direction
        for f in (fl + 1)..8 {
            result |= 1 << (f + rk * 8);
            if block & (1 << (f + rk * 8)) != 0 {
                break;
            }
        }

        // Negative file direction
        for f in (0..fl).rev() {
            result |= 1 << (f + rk * 8);
            if block & (1 << (f + rk * 8)) != 0 {
                break;
            }
        }

        result
    }

    fn batt(sq: i32, block: u64) -> u64 {
        let mut result: u64 = 0;
        let rk = sq / 8; // Rank
        let fl = sq % 8; // File

        // Diagonal: bottom left to top right
        let mut r = rk + 1;
        let mut f = fl + 1;
        while r < 8 && f < 8 {
            result |= 1 << (f + r * 8);
            if block & (1 << (f + r * 8)) != 0 {
                break;
            }
            r += 1;
            f += 1;
        }

        // Diagonal: top left to bottom right
        r = rk + 1;
        f = fl - 1;
        while r < 8 && f >= 0 {
            result |= 1 << (f + r * 8);
            if block & (1 << (f + r * 8)) != 0 {
                break;
            }
            r += 1;
            f -= 1;
        }

        // Diagonal: top right to bottom left
        r = rk - 1;
        f = fl + 1;
        while r >= 0 && f < 8 {
            result |= 1 << (f + r * 8);
            if block & (1 << (f + r * 8)) != 0 {
                break;
            }
            r -= 1;
            f += 1;
        }

        // Diagonal: bottom right to top left
        r = rk - 1;
        f = fl - 1;
        while r >= 0 && f >= 0 {
            result |= 1 << (f + r * 8);
            if block & (1 << (f + r * 8)) != 0 {
                break;
            }
            r -= 1;
            f -= 1;
        }

        result
    }

    fn random_uint64() -> u64 {
        let mut rng = rand::thread_rng();
        let u1: u64 = rng.gen::<u16>() as u64;
        let u2: u64 = rng.gen::<u16>() as u64;
        let u3: u64 = rng.gen::<u16>() as u64;
        let u4: u64 = rng.gen::<u16>() as u64;
        u1 | (u2 << 16) | (u3 << 32) | (u4 << 48)
    }

    fn random_uint64_fewbits() -> u64 {
        Self::random_uint64() & Self::random_uint64() & Self::random_uint64()
    }

    /// Finds a suitable magic number for the given square and mask.
    pub fn find_magic(sq: i32, m: i32, bishop: bool) -> u64 {
        let mask = if bishop {
            BISHOP_MASKS[sq as usize]
        } else {
            ROOK_MASKS[sq as usize]
        };
        let n = mask.0.count_ones();
        let mut b = vec![0; 1 << n];
        let mut a = vec![0; 1 << n];
        let mut used = vec![0; 1 << n];

        for i in 0..(1 << n) {
            b[i] = Self::index_to_uint64(i as i32, n.try_into().unwrap(), mask.0);
            a[i] = if bishop {
                Self::batt(sq, b[i])
            } else {
                Self::ratt(sq, b[i])
            };
        }

        for _ in 0..100_000_000 {
            let magic = Self::random_uint64_fewbits();
            if ((mask.0.wrapping_mul(magic)) & 0xFF00000000000000).count_ones() < 6 {
                continue;
            }

            used.iter_mut().for_each(|x| *x = 0);

            let mut fail = false;
            for i in 0..(1 << n) {
                let j = Self::transform(b[i], magic, m) as usize;
                if used[j] == 0 {
                    used[j] = a[i];
                } else if used[j] != a[i] {
                    fail = true;
                    break;
                }
            }

            if !fail {
                return magic;
            }
        }

        panic!("Failed to find a magic number");
    }
}
