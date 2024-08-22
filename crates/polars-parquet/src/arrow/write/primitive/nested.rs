use arrow::array::{Array, PrimitiveArray};
use arrow::types::NativeType as ArrowNativeType;
use polars_error::PolarsResult;

use super::super::{nested, utils, WriteOptions};
use super::basic::{build_statistics, encode_plain};
use crate::arrow::read::schema::is_nullable;
use crate::arrow::write::Nested;
use crate::parquet::encoding::Encoding;
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::types::NativeType;
use crate::write::primitive::ArrayContext;

pub fn array_to_page<T, R>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
    nested: &[Nested],
) -> PolarsResult<DataPage>
where
    T: ArrowNativeType,
    R: NativeType,
    T: num_traits::AsPrimitive<R>,
{
    let materialize_nulls = Nested::do_materialize_nulls(&nested);
    let is_optional = is_nullable(&type_.field_info);

    let ctx = ArrayContext::new(is_optional, materialize_nulls);

    let mut buffer = vec![];

    let (repetition_levels_byte_length, definition_levels_byte_length) =
        nested::write_rep_and_def(options.version, nested, &mut buffer)?;

    dbg!(ctx);
    dbg!(array);

    let prev_len = buffer.len();
    let buffer = encode_plain(array, ctx, buffer);

    dbg!(&buffer[prev_len..]);

    let statistics = if options.has_statistics() {
        Some(build_statistics(array, type_.clone(), &options.statistics).serialize())
    } else {
        None
    };

    utils::build_plain_page(
        buffer,
        dbg!(nested::num_values(nested)),
        nested[0].len(),
        array.null_count(),
        repetition_levels_byte_length,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        Encoding::Plain,
    )
}
