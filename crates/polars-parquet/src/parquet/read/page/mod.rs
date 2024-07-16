mod indexed_reader;
mod reader;
#[cfg(feature = "async")]
mod stream;

pub use indexed_reader::IndexedPageReader;
pub use reader::{PageFilter, PageMetaData, PageReader};

use crate::parquet::error::ParquetError;
use crate::parquet::page::CompressedPage;

pub trait PageIterator<'a>: Iterator<Item = Result<CompressedPage<'a>, ParquetError>> {
    fn swap_buffer(&mut self, buffer: &mut Vec<u8>);
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use stream::{get_page_stream, get_page_stream_from_column_start};
