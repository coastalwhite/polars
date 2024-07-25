use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::dictionary::*;
use super::super::nested_utils::{InitNested, NestedState};
use super::super::utils::MaybeNext;
use super::basic::deserialize_plain;
use super::DecoderFunction;
use crate::parquet::page::DictPage;
use crate::parquet::read::BasicDecompressor;
use crate::parquet::types::NativeType as ParquetNativeType;
use crate::read::CompressedPagesIter;

fn read_dict<P, T, D>(data_type: ArrowDataType, dict: &DictPage, decoder: D) -> Box<dyn Array>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    let data_type = match data_type {
        ArrowDataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };
    let values = deserialize_plain::<P, T, D>(&dict.buffer, decoder);
    Box::new(PrimitiveArray::new(data_type, values.into(), None))
}

/// An iterator adapter that converts [`DataPages`] into an [`Iterator`] of [`DictionaryArray`]
pub struct NestedDictIter<K, T, I, P, D>
where
    I: CompressedPagesIter,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    iter: BasicDecompressor<I>,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
    decoder: D,
    phantom: std::marker::PhantomData<(P, T)>,
}

impl<K, T, I, P, D> NestedDictIter<K, T, I, P, D>
where
    K: DictionaryKey,
    I: CompressedPagesIter,
    T: NativeType,

    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    pub fn new(
        iter: BasicDecompressor<I>,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        decoder: D,
    ) -> Self {
        Self {
            iter,
            init,
            data_type,
            values: None,
            items: VecDeque::new(),
            remaining: num_rows,
            chunk_size,
            decoder,
            phantom: Default::default(),
        }
    }
}

impl<K, T, I, P, D> Iterator for NestedDictIter<K, T, I, P, D>
where
    I: CompressedPagesIter,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type Item = PolarsResult<(NestedState, DictionaryArray<K>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = nested_next_dict(
                &mut self.iter,
                &mut self.items,
                &mut self.remaining,
                &self.init,
                &mut self.values,
                self.data_type.clone(),
                self.chunk_size,
                |dict| read_dict(self.data_type.clone(), dict, self.decoder),
            );
            match maybe_state {
                MaybeNext::Some(Ok(dict)) => return Some(Ok(dict)),
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
