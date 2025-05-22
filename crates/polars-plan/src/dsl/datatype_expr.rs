use core::fmt;

use polars_core::error::{PolarsResult, polars_bail};
use polars_core::prelude::DataType;
use polars_core::schema::Schema;
use polars_core::utils::try_get_supertype;
use polars_utils::arena::Arena;

use super::Expr;
use crate::plans::to_expr_ir;

#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DataTypeExpr {
    Literal(DataType),
    OfExpr(Box<Expr>),
    Supertype { dtypes: Vec<DataTypeExpr> },
}

#[recursive::recursive]
fn into_datatype_impl(dt_expr: DataTypeExpr, schema: &Schema) -> PolarsResult<DataType> {
    Ok(match dt_expr {
        DataTypeExpr::Literal(dt) => dt,
        DataTypeExpr::OfExpr(expr) => {
            let mut arena = Arena::new();
            let e = to_expr_ir(*expr, &mut arena, schema)?;
            arena
                .get(e.node())
                .to_dtype(schema, Default::default(), &arena)?
        },
        DataTypeExpr::Supertype { dtypes } => {
            let mut dtypes = dtypes.into_iter();
            let Some(e) = dtypes.next() else {
                polars_bail!(InvalidOperation: "cannot take supertype of no datatype expressions");
            };
            let mut dtype = e.into_datatype(schema)?;
            for e in dtypes {
                dtype = try_get_supertype(&dtype, &e.into_datatype(schema)?)?;
            }
            dtype
        },
    })
}

impl DataTypeExpr {
    pub fn into_datatype(self, schema: &Schema) -> PolarsResult<DataType> {
        into_datatype_impl(self, schema)
    }

    pub fn expr_inputs_rev<'a>(&'a self, inputs: &'_ mut Vec<&'a Expr>) {
        match self {
            Self::Literal(_) => {},
            Self::OfExpr(expr) => inputs.push(expr.as_ref()),
            Self::Supertype { dtypes } => {
                for dtype in dtypes {
                    dtype.expr_inputs_rev(inputs);
                }
            },
        }
    }

    pub fn as_literal(&self) -> Option<&DataType> {
        match self {
            Self::Literal(dt) => Some(dt),
            _ => None,
        }
    }
}

impl fmt::Debug for DataTypeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataTypeExpr::Literal(data_type) => data_type.fmt(f),
            DataTypeExpr::OfExpr(expr) => write!(f, "dtype_of({expr:?})"),
            DataTypeExpr::Supertype { dtypes } => {
                f.write_str("supertype_of(")?;
                if let Some(fst) = dtypes.first() {
                    fst.fmt(f)?;
                    for dtype in &dtypes[1..] {
                        f.write_str(", ")?;
                        dtype.fmt(f)?;
                    }
                }
                f.write_str(")")?;
                Ok(())
            },
        }
    }
}

impl From<DataType> for DataTypeExpr {
    fn from(value: DataType) -> Self {
        Self::Literal(value)
    }
}
