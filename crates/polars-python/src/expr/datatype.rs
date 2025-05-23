use polars::prelude::{DataType, DataTypeExpr};
use pyo3::pyclass;

use crate::prelude::Wrap;

use super::PyExpr;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataTypeExpr {
    pub inner: DataTypeExpr,
}

impl From<DataTypeExpr> for PyDataTypeExpr {
    fn from(expr: DataTypeExpr) -> Self {
        PyDataTypeExpr { inner: expr }
    }
}

#[cfg(feature = "pymethods")]
#[pyo3::pymethods]
impl PyDataTypeExpr {
    #[staticmethod]
    pub fn from_dtype(datatype: Wrap<DataType>) -> Self {
        DataTypeExpr::Literal(datatype.0).into()
    }

    #[staticmethod]
    pub fn of_expr(expr: PyExpr) -> Self {
        DataTypeExpr::OfExpr(Box::new(expr.inner)).into()
    }

    #[staticmethod]
    pub fn supertype_of(exprs: Vec<PyDataTypeExpr>) -> Self {
        DataTypeExpr::Supertype {
            dtypes: exprs.into_iter().map(|i| i.inner).collect(),
        }.into()
    }

}
