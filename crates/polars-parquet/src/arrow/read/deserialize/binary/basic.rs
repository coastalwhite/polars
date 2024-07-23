use std::default::Default;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::specification::try_check_utf8;
use arrow::array::Array;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offset;
use polars_error::PolarsResult;
use polars_utils::iter::FallibleIterator;

use super::super::utils;
use super::super::utils::{extend_from_decoder, DecodedState};
use super::decoders::*;
use super::utils::*;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage};
use crate::read::deserialize::utils::StateTranslation;
use crate::read::PrimitiveLogicalType;

impl<O: Offset> DecodedState for (Binary<O>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, O: Offset> StateTranslation<'a, BinaryDecoder<O>> for BinaryStateTranslation<'a> {
    fn new(
        decoder: &BinaryDecoder<O>,
        page: &'a DataPage,
        dict: Option<&'a <BinaryDecoder<O> as utils::Decoder>::Dict>,
        page_validity: Option<&utils::PageValidity<'a>>,
        filter: Option<&utils::filter::Filter<'a>>,
    ) -> PolarsResult<Self> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        decoder.check_utf8.store(is_string, Ordering::Relaxed);
        BinaryStateTranslation::new(page, dict, page_validity, filter, is_string)
    }

    fn len_when_not_nullable(&self) -> usize {
        BinaryStateTranslation::len_when_not_nullable(self)
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        BinaryStateTranslation::skip_in_place(self, n)
    }

    fn extend_from_state(
        &mut self,
        decoder: &BinaryDecoder<O>,
        decoded: &mut <BinaryDecoder<O> as utils::Decoder>::DecodedState,
        page_validity: &mut Option<utils::PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        let mut validate_utf8 = decoder.check_utf8.load(Ordering::Relaxed);
        let len_before = values.offsets.len();

        use BinaryStateTranslation as T;
        match (self, page_validity) {
            (T::Plain(page_values), None) => {
                for x in page_values.by_ref().take(additional) {
                    values.push(x)
                }
            },
            (T::Plain(page_values), Some(page_validity)) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            )?,
            (T::Dictionary(page, _), None) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page.dict;

                for x in page
                    .values
                    .by_ref()
                    .map(|index| page_dict.value(index as usize))
                    .take(additional)
                {
                    values.push(x)
                }
                page.values.get_result()?;
            },
            (T::Dictionary(page, _), Some(page_validity)) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page.dict;
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut page
                        .values
                        .by_ref()
                        .map(|index| page_dict.value(index as usize)),
                )?;
                page.values.get_result()?;
            },
            (T::Delta(page), None) => {
                values.extend_lengths(page.lengths.by_ref().take(additional), &mut page.values);
            },
            (T::Delta(page), Some(page_validity)) => {
                let Binary {
                    offsets,
                    values: values_,
                } = values;

                let last_offset = *offsets.last();
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    offsets,
                    page.lengths.by_ref(),
                )?;

                let length = *offsets.last() - last_offset;

                let (consumed, remaining) = page.values.split_at(length.to_usize());
                page.values = remaining;
                values_.extend_from_slice(consumed);
            },
            (T::DeltaBytes(page_values), None) => {
                for x in page_values.take(additional) {
                    values.push(x)
                }
            },
            (T::DeltaBytes(page_values), Some(page_validity)) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            )?,
        }

        if validate_utf8 {
            // @TODO: This can report a better error.
            let offsets = &values.offsets.as_slice()[len_before..];
            try_check_utf8(offsets, &values.values).map_err(|_| ParquetError::oos("invalid utf-8"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BinaryDecoder<O: Offset> {
    phantom_o: std::marker::PhantomData<O>,
    check_utf8: AtomicBool,
}

impl<O: Offset> utils::Decoder for BinaryDecoder<O> {
    type Translation<'a> = BinaryStateTranslation<'a>;
    type Dict = BinaryDict;
    type DecodedState = (Binary<O>, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Binary::<O>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>> {
        super::finalize(data_type, values, validity)
    }
}
