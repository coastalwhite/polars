use arrow::array::{
    Array, BinaryViewArray, BooleanArray, ListArray, NullArray, PrimitiveArray, StructArray,
    Utf8ViewArray,
};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::Offset;

use super::TotalOrdKernel;

macro_rules! call_binary {
    ($T:ty, $lhs:expr, $rhs:expr, $wrong_width:expr, $invalid:expr, $op:expr) => {{
        let mut bitmap = MutableBitmap::with_capacity($lhs.len());

        for i in 0..$lhs.len() {
            let lhs_validity = $lhs.validity().map_or(true, |v| v.get(i).unwrap());
            let rhs_validity = $rhs.validity().map_or(true, |v| v.get(i).unwrap());
            let (lhs_start, lhs_end) = $lhs.offsets().start_end(i);
            let (rhs_start, rhs_end) = $rhs.offsets().start_end(i);

            if !lhs_validity || !rhs_validity {
                bitmap.push($invalid);
                continue;
            }

            if lhs_end - lhs_start != rhs_end - rhs_start {
                bitmap.push($wrong_width);
                continue;
            }

            let mut lhs = $lhs.clone();
            lhs.slice(lhs_start, lhs_end - lhs_start);
            let lhs: &$T = lhs.as_any().downcast_ref().unwrap();
            let mut rhs = $rhs.clone();
            rhs.slice(rhs_start, rhs_end - rhs_start);
            let rhs: &$T = rhs.as_any().downcast_ref().unwrap();

            bitmap.push($op(lhs, rhs));
        }

        bitmap.freeze()
    }};
}

macro_rules! compare {
    ($lhs:expr, $rhs:expr, $wrong_width:expr, $invalid:expr, $op:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        assert_eq!(lhs.len(), rhs.len());
        let ArrowDataType::List(lhs_type) = lhs.data_type().to_logical_type() else {
            panic!("array comparison called with non-array type");
        };
        let ArrowDataType::List(rhs_type) = rhs.data_type().to_logical_type() else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(lhs_type.data_type(), rhs_type.data_type());

        use arrow::datatypes::{PhysicalType as PH, PrimitiveType as PR};
        match lhs_type.data_type().to_physical_type() {
            PH::Boolean => {
                call_binary!(BooleanArray, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::BinaryView => call_binary!(BinaryViewArray, lhs, rhs, $wrong_width, $invalid, $op),
            PH::Utf8View => {
                call_binary!(Utf8ViewArray, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::Int8) => {
                call_binary!(PrimitiveArray<i8>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::Int16) => {
                call_binary!(PrimitiveArray<i16>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::Int32) => {
                call_binary!(PrimitiveArray<i32>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::Int64) => {
                call_binary!(PrimitiveArray<i64>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::Int128) => {
                call_binary!(PrimitiveArray<i128>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::UInt8) => {
                call_binary!(PrimitiveArray<u8>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::UInt16) => {
                call_binary!(PrimitiveArray<u16>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::UInt32) => {
                call_binary!(PrimitiveArray<u32>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::UInt64) => {
                call_binary!(PrimitiveArray<u64>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::UInt128) => {
                call_binary!(PrimitiveArray<u128>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::Float16) => {
                todo!("Comparison of List with Primitive(Float16) are not yet supported")
            },
            PH::Primitive(PR::Float32) => {
                call_binary!(PrimitiveArray<f32>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::Float64) => {
                call_binary!(PrimitiveArray<f64>, lhs, rhs, $wrong_width, $invalid, $op)
            },
            PH::Primitive(PR::Int256) => {
                todo!("Comparison of List with Primitive(Int256) are not yet supported")
            },
            PH::Primitive(PR::DaysMs) => {
                todo!("Comparison of List with Primitive(DaysMs) are not yet supported")
            },
            PH::Primitive(PR::MonthDayNano) => {
                todo!("Comparison of List with Primitive(MonthDayNano) are not yet supported")
            },

            #[cfg(feature = "dtype-array")]
            PH::FixedSizeList => call_binary!(arrow::array::FixedSizeListArray, lhs, rhs, $wrong_width, $invalid, $op),
            #[cfg(not(feature = "dtype-array"))]
            PH::FixedSizeList => todo!("Comparison of List with FixedSizeList are not supported without the `dtype-array` feature"),

            PH::Null => call_binary!(NullArray, lhs, rhs, $wrong_width, $invalid, $op),
            PH::Binary => todo!("Comparison of List with Binary are not yet supported"),
            PH::FixedSizeBinary => {
                todo!("Comparison of List with FixedSizeBinary are not yet supported")
            },
            PH::LargeBinary => todo!("Comparison of List with LargeBinary are not yet supported"),
            PH::Utf8 => todo!("Comparison of List with Utf8 are not yet supported"),
            PH::LargeUtf8 => todo!("Comparison of List with LargeUtf8 are not yet supported"),
            PH::List => call_binary!(ListArray<i32>, lhs, rhs, $wrong_width, $invalid, $op),
            PH::LargeList => call_binary!(ListArray<i64>, lhs, rhs, $wrong_width, $invalid, $op),
            PH::Struct => call_binary!(StructArray, lhs, rhs, $wrong_width, $invalid, $op),
            PH::Union => todo!("Comparison of List with Union are not yet supported"),
            PH::Map => todo!("Comparison of List with Map are not yet supported"),
            PH::Dictionary(_) => {
                todo!("Comparison of List with Dictionary are not yet supported")
            },
        }
    }};
}

impl<O: Offset> TotalOrdKernel for ListArray<O> {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        compare!(
            self,
            other,
            false,
            true,
            |lhs, rhs| TotalOrdKernel::tot_eq_missing_kernel(lhs, rhs).unset_bits() == 0
        )
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        compare!(
            self,
            other,
            true,
            true,
            |lhs, rhs| TotalOrdKernel::tot_ne_missing_kernel(lhs, rhs).unset_bits() == 0
        )
    }

    fn tot_lt_kernel(&self, _other: &Self) -> Bitmap {
        unimplemented!()
    }

    fn tot_le_kernel(&self, _other: &Self) -> Bitmap {
        unimplemented!()
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }

    fn tot_lt_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_le_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_gt_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_ge_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }
}
