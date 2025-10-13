use arrow::array::{Array, BinaryViewArray, BooleanArray, PrimitiveArray, StaticArray};
use arrow::datatypes::ArrowDataType;
use arrow::legacy::prelude::LargeBinaryArray;
use arrow::types::{NativeType, PrimitiveType, f16};
use polars_utils::aliases::PlHashSet;
use polars_utils::total_ord::{TotalEq, TotalHash, TotalOrdWrap};
use polars_utils::{IdxSize, UnitVec};

pub trait AmortizedUnique: Send + Sync + 'static {
    fn new_empty(&self) -> Box<dyn AmortizedUnique>;
    fn arg_unique(
        &mut self,
        values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    );
    fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>);
}

pub fn amortized_unique_from_dtype(
    dtype: &ArrowDataType,
) -> Box<dyn AmortizedUnique> {
    use arrow::datatypes::PhysicalType as P;
    match dtype.to_physical_type() {
        P::Null => Box::new(NullUnique) as _,
        P::Boolean => Box::new(BooleanUnique) as _,
        P::Primitive(pt) => match pt {
            PrimitiveType::Int8 => Box::new(PrimitiveArgUnique::<i8>(Default::default())) as _,
            PrimitiveType::Int16 => Box::new(PrimitiveArgUnique::<i16>(Default::default())) as _,
            PrimitiveType::Int32 => Box::new(PrimitiveArgUnique::<i32>(Default::default())) as _,
            PrimitiveType::Int64 => Box::new(PrimitiveArgUnique::<i64>(Default::default())) as _,
            PrimitiveType::Int128 => Box::new(PrimitiveArgUnique::<i128>(Default::default())) as _,
            PrimitiveType::UInt8 => Box::new(PrimitiveArgUnique::<u8>(Default::default())) as _,
            PrimitiveType::UInt16 => Box::new(PrimitiveArgUnique::<u16>(Default::default())) as _,
            PrimitiveType::UInt32 => Box::new(PrimitiveArgUnique::<u32>(Default::default())) as _,
            PrimitiveType::UInt64 => Box::new(PrimitiveArgUnique::<u64>(Default::default())) as _,
            PrimitiveType::UInt128 => Box::new(PrimitiveArgUnique::<u128>(Default::default())) as _,
            PrimitiveType::Float16 => Box::new(FloatArgUnique::<f16>(Default::default())) as _,
            PrimitiveType::Float32 => Box::new(FloatArgUnique::<f32>(Default::default())) as _,
            PrimitiveType::Float64 => Box::new(FloatArgUnique::<f64>(Default::default())) as _,
            PrimitiveType::Int256 => unreachable!(),
            PrimitiveType::DaysMs => unreachable!(),
            PrimitiveType::MonthDayNano => unreachable!(),
            PrimitiveType::MonthDayMillis => unreachable!(),
        },
        P::BinaryView => Box::new(BinaryViewUnique(Default::default())) as _,
        P::LargeBinary => Box::new(BinaryUnique(Default::default())) as _,

        P::Dictionary(_) => unreachable!(),
        P::Binary => unreachable!(),
        P::FixedSizeBinary => unreachable!(),
        P::Utf8 => unreachable!(),
        P::LargeUtf8 => unreachable!(),
        P::List => unreachable!(),
        P::Union => unreachable!(),
        P::Map => unreachable!(),

        // Should be handled through BinaryView.
        P::Utf8View => unreachable!(),

        // Should be handled through row encoding.
        P::FixedSizeList => unreachable!(),
        P::LargeList => unreachable!(),
        P::Struct => unreachable!(),
    }
}

struct NullUnique;
struct BooleanUnique;
struct FloatArgUnique<T>(PlHashSet<Option<TotalOrdWrap<T>>>);
struct PrimitiveArgUnique<T>(PlHashSet<Option<T>>);
struct BinaryViewUnique(PlHashSet<Option<&'static [u8]>>);
struct BinaryUnique(PlHashSet<Option<&'static [u8]>>);

impl AmortizedUnique for NullUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(NullUnique)
    }

    fn retain_unique(&mut self, _values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        let mut is_first = true;
        idxs.retain(|_| std::mem::replace(&mut is_first, false));
    }

    fn arg_unique(
        &mut self,
        _values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        if length > 0 {
            idxs.push(start);
        }
    }
}

impl AmortizedUnique for BooleanUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(BooleanUnique)
    }

    fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        let mut seen = 0u8;
        let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
        idxs.retain(|i| {
            let v = match values.get(*i as usize) {
                None => 0,
                Some(false) => 1,
                Some(true) => 2,
            };

            if seen & v != 0 {
                false
            } else {
                seen |= v;
                true
            }
        });
    }

    fn arg_unique(
        &mut self,
        values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        let mut seen = 0u8;
        let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
        idxs.extend((start..start + length).filter_map(|i| {
            let v = match values.get(i as usize) {
                None => 0,
                Some(false) => 1,
                Some(true) => 2,
            };

            if seen & v != 0 {
                None
            } else {
                seen |= v;
                Some(i)
            }
        }));
    }
}


impl<T: NativeType + TotalHash + TotalEq> AmortizedUnique for FloatArgUnique<T> {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(FloatArgUnique::<T>(Default::default()))
    }

    fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        self.0.clear();
        let values = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        idxs.retain(|&i| self.0.insert(values.get(i as usize).map(TotalOrdWrap)));
    }

    fn arg_unique(
        &mut self,
        values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        self.0.clear();
        let values = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        idxs.extend((start..start + length).filter_map(|i| {
            self.0
                .insert(values.get(i as usize).map(TotalOrdWrap))
                .then_some(i)
        }));
    }
}


impl<T: NativeType + std::hash::Hash + Eq> AmortizedUnique for PrimitiveArgUnique<T> {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(PrimitiveArgUnique::<T>(Default::default()))
    }

    fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        self.0.clear();
        let values = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        idxs.retain(|&i| self.0.insert(values.get(i as usize)));
    }

    fn arg_unique(
        &mut self,
        values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        self.0.clear();
        let values = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        idxs.extend(
            (start..start + length)
                .filter_map(|i| self.0.insert(values.get(i as usize)).then_some(i)),
        );
    }
}


impl AmortizedUnique for BinaryViewUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(BinaryViewUnique(Default::default()))
    }

    fn arg_unique(
        &mut self,
        values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        let values = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
        idxs.extend((start..start + length).filter_map(|i| {
            self.0
                .insert(
                    values
                        .get(i as usize)
                        .map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) }),
                )
                .then_some(i)
        }));
        self.0.clear();
    }

    fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        let values = values.as_any().downcast_ref::<BinaryViewArray>().unwrap();
        idxs.retain(|&i| {
            self.0.insert(
                values
                    .get(i as usize)
                    .map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) }),
            )
        });
        self.0.clear();
    }
}

impl AmortizedUnique for BinaryUnique {
    fn new_empty(&self) -> Box<dyn AmortizedUnique> {
        Box::new(BinaryUnique(Default::default()))
    }

    fn arg_unique(
        &mut self,
        values: &dyn Array,
        idxs: &mut UnitVec<IdxSize>,
        start: IdxSize,
        length: IdxSize,
    ) {
        let values = values.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        idxs.extend((start..start + length).filter_map(|i| {
            self.0
                .insert(
                    values
                        .get(i as usize)
                        .map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) }),
                )
                .then_some(i)
        }));
        self.0.clear();
    }

    fn retain_unique(&mut self, values: &dyn Array, idxs: &mut UnitVec<IdxSize>) {
        let values = values.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        idxs.retain(|&i| {
            self.0.insert(
                values
                    .get(i as usize)
                    .map(|v| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(v) }),
            )
        });
        self.0.clear();
    }
}
