pub trait SelectModulo {
    fn select_modulo(self, m: u32) -> Option<u8>;
}

macro_rules! impl_select_modulo {
    ($($t:ty),+) => {
        $(
        impl SelectModulo for $t {
            fn select_modulo(self, m: u32) -> Option<u8> {
                if self == 0 {
                    return None;
                }

                let mut c = self;
                let mut offset = 0u8;
                let mut selected = m % c.count_ones();

                loop {
                    let shift = if c & 1 == 0 {
                        c.trailing_zeros()
                    } else {
                        0
                    };

                    c >>= shift;
                    offset += shift as u8;

                    let trailing_ones = c.trailing_ones();

                    if selected < trailing_ones {
                        offset += selected as u8;
                        break;
                    }

                    let shift = trailing_ones;
                    c >>= shift;
                    offset += shift as u8;
                    selected -= shift;
                }

                Some(offset)
            }
        }
        )+
    };
}

impl_select_modulo!(u8, u16, u32, u64, u128);

#[test]
fn test_select_modulo() {
    assert_eq!(0u32.select_modulo(0), None);
    assert_eq!(0u32.select_modulo(1), None);

    assert_eq!(0x11u32.select_modulo(0), Some(0));
    assert_eq!(0x11u32.select_modulo(1), Some(4));
    assert_eq!(0x11u32.select_modulo(2132131), Some(4));
    assert_eq!(0xFFu32.select_modulo(0x2132131), Some(1));
    assert_eq!(0xFFu32.select_modulo(0x2132137), Some(7));
    assert_eq!(0xF00Fu32.select_modulo(0x2132137), Some(15));
}
