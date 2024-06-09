macro_rules! bitset {
    (
    $(#[$($set_attrss:tt)*])*
    $set_vis:vis struct $set:ident,

    $(#[$($flags_attrss:tt)*])*
    $flags_vis:vis enum $flags:ident: $base:ty {
        $(
        $(#[$($item_attrss:tt)*])*
        $item:ident,
        )+
    }
    )=> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        $(#[$($set_attrss)*])*
        #[repr(transparent)]
        $set_vis struct $set($base);

        #[derive(Clone, Copy, PartialEq, Eq)]
        $(#[$($flags_attrss)*])*
        #[repr(u8)]
        $flags_vis enum $flags {
            $(
            $(#[$($item_attrss)*])*
            $item,
            )+
        }

        impl $set {
            /// Create a empty set
            #[allow(unused)]
            pub const fn empty() -> Self {
                Self(0)
            }

            /// Create a set containing all items
            #[allow(unused)]
            pub const fn all() -> Self {
                Self(
                    $(
                    $flags::$item.to_mask() |
                    )+
                    0
                )
            }

            #[allow(unused)]
            pub const fn contains_flag(self, flag: $flags) -> bool {
                self.0 & flag.to_mask() != 0
            }

            #[allow(unused)]
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 != other.0
            }

            #[allow(unused)]
            pub fn select_modulo(self, m: u32) -> Option<$flags> {
                let value = <$base as crate::select_modulo::SelectModulo>::select_modulo(self.0, m)?;
                $flags::try_from(value).ok()
            }
        }

        impl $flags {
            const fn to_mask(self) -> $base {
                1 << (self as u8)
            }
        }

        impl ::std::ops::BitAnd<Self> for $set {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                Self(self.0 & rhs.0)
            }
        }

        impl ::std::ops::BitAndAssign<Self> for $set {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl ::std::ops::BitOr<Self> for $set {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                Self(self.0 | rhs.0)
            }
        }

        impl ::std::ops::BitOrAssign<Self> for $set {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl ::std::ops::BitOr<$flags> for $set {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: $flags) -> Self::Output {
                Self(self.0 | rhs.to_mask())
            }
        }

        impl ::std::ops::BitOr<$set> for $flags {
            type Output = $set;

            #[inline]
            fn bitor(self, rhs: $set) -> Self::Output {
                $set(self.to_mask() | rhs.0)
            }
        }

        impl ::std::ops::BitOr<$flags> for $flags {
            type Output = $set;

            #[inline]
            fn bitor(self, rhs: $flags) -> Self::Output {
                $set(self.to_mask() | rhs.to_mask())
            }
        }

        impl ::std::convert::TryFrom<u8> for $flags {
            type Error = ();

            #[inline]
            fn try_from(v: u8) -> Result<Self, Self::Error> {
                let mut i = 0;
                #[allow(unused_assignments)]
                match v {
                    $(_ if v == { let x = i; i += 1; x } => Ok(Self::$item),)+
                    _ => Err(()),
                }
            }
        }
    };
}

bitset! {
    pub struct Set, pub enum Flags: u64 {
        A,
        B,
        C,
    }
}
