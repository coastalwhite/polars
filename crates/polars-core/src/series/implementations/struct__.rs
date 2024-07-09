use std::ops::Not;
use arrow::bitmap::Bitmap;
use crate::chunked_array::StructChunked2;
use super::*;
use crate::hashing::series_to_hashes;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};


impl PrivateSeriesNumeric for SeriesWrap<StructChunked2> {
    fn bit_repr(&self) -> Option<BitRepr> {
        None
    }
}

impl PrivateSeries for SeriesWrap<StructChunked2> {
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn compute_len(&mut self) {
        self.0.compute_len()
    }

    fn _get_flags(&self) -> MetadataFlags {
        MetadataFlags::empty()
    }

    fn _set_flags(&mut self, _flags: MetadataFlags) {}

    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self._apply_fields(|s|s.explode_by_offsets(offsets)).unwrap().into_series()
    }

    // TODO! remove this. Very slow. Asof join should use row-encoding.
    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let other = other.struct_().unwrap();
        self.0
            .fields_as_series()
            .iter()
            .zip(other.fields_as_series())
            .all(|(s, other)| s.equal_element(idx_self, idx_other, &other))
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        let other = other.struct_()?;
        let fields = self
            .0
            .fields_as_series()
            .iter()
            .zip(other.fields_as_series())
            .map(|(lhs, rhs)| lhs.zip_with_same_type(mask, &rhs))
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(StructChunked::new_unchecked(self.0.name(), &fields).into_series())
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_list(groups)
    }

}

impl SeriesTrait for SeriesWrap<StructChunked2> {
    fn rename(&mut self, name: &str) {
        self.0.rename(name)
    }

    fn chunk_lengths(&self) -> ChunkLenIter {
        self.0.chunk_lengths()
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        &self.0.chunks
    }

    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.chunks_mut()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (l, r) = self.0.split_at(offset);
        (l.into_series(), r.into_series())
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        self.0.append(other.as_ref().as_ref());
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        self.0.extend(other.as_ref().as_ref());
        Ok(())
    }

    fn filter(&self, _filter: &BooleanChunked) -> PolarsResult<Series> {
        ChunkFilter::filter(&self.0, _filter).map(|ca| ca.into_series())
    }

    fn take(&self, _indices: &IdxCa) -> PolarsResult<Series> {
        self.0.take(_indices).map(|ca| ca.into_series())
    }

    unsafe fn take_unchecked(&self, _idx: &IdxCa) -> Series {
        self.0.take_unchecked(_idx).into_series()
    }

    fn take_slice(&self, _indices: &[IdxSize]) -> PolarsResult<Series> {
        self.0.take(_indices).map(|ca|ca.into_series())
    }

    unsafe fn take_slice_unchecked(&self, _idx: &[IdxSize]) -> Series {
        self.0.take_unchecked(_idx).into_series()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        let ca = self.0.rechunk();
        ca.into_series()
    }

    fn new_from_index(&self, _index: usize, _length: usize) -> Series {
        self.0.new_from_index(_length, _index).into_series()
    }

    fn cast(&self, dtype: &DataType, cast_options: CastOptions) -> PolarsResult<Series> {
        self.0.cast_with_options(dtype, cast_options)
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        self.0.get_any_value(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
    }

    fn is_null(&self) -> BooleanChunked {
        let iter = self.downcast_iter().map(|arr| {
            let bitmap = match arr.validity() {
                Some(valid) => valid.not(),
                None => Bitmap::new_with_value(false, arr.len())
            };
            BooleanArray::from_data_default(bitmap, None)
        });
        BooleanChunked::from_chunk_iter(self.name(), iter)
    }

    fn is_not_null(&self) -> BooleanChunked {
        let iter = self.downcast_iter().map(|arr| {
            let bitmap = match arr.validity() {
                Some(valid) => valid.clone(),
                None => Bitmap::new_with_value(true, arr.len())
            };
            BooleanArray::from_data_default(bitmap, None)
        });
        BooleanChunked::from_chunk_iter(self.name(), iter)
    }

    fn reverse(&self) -> Series {
        self.0._apply_fields(|s| s.reverse()).unwrap().into_series()
    }

    fn shift(&self, periods: i64) -> Series {
        self.0._apply_fields(|s| s.shift(periods)).unwrap().into_series()
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
        Ok(self.0.sort_with(options).into_series())
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }
}
