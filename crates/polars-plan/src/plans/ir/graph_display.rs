use std::fmt;

use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_ops::frame::JoinType;
use polars_tpg::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::IR;
use crate::dsl::{FileScanIR, SinkTypeIR};
use crate::plans::{AExpr, IRPlan};

pub fn to_tpg(root: &[Node], ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> TextPlanGraph {
    let mut next_key = 0;
    let mut nodes = Vec::with_capacity(16);
    let mut stack: Vec<(TpgKey, Node)> = Vec::with_capacity(16);
    let mut visited: PlHashMap<Node, TpgKey> = PlHashMap::with_capacity(16);

    macro_rules! d {
        ($expr:expr) => {{ $expr.display(expr_arena).to_string() }};
    }
    macro_rules! ds {
        ($exprs:expr) => {{ TpgListBuilder::new().items($exprs.iter().map(|e| d!(e))) }};
    }

    macro_rules! new_child {
        ($child:expr) => {{
            let child: Node = $child;
            if matches!(ir_arena.get($child), IR::Cache { .. })
                && let Some(graph_key) = visited.get(&child)
            {
                *graph_key
            } else {
                let graph_key = next_key;
                next_key += 1;
                let graph_key = TpgKey(graph_key);
                visited.insert(child, graph_key);
                stack.push((graph_key, child));
                graph_key
            }
        }};
    }

    let mut roots = 0;
    for &n in root {
        match ir_arena.get(n) {
            IR::SinkMultiple { inputs } => {
                roots += inputs.len();
                for n in inputs.iter() {
                    new_child!(*n);
                }
            },
            _ => {
                roots += 1;
                new_child!(n);
            },
        }
    }

    let mut inputs_scratch = Vec::with_capacity(4);
    while let Some((graph_key, node)) = stack.pop() {
        let ir = ir_arena.get(node);

        inputs_scratch.clear();
        ir.copy_inputs(&mut inputs_scratch);
        let children: Vec<TpgKey> = inputs_scratch
            .iter()
            .copied()
            .map(|n| new_child!(n))
            .collect();

        use {TpgNode as C, TpgVerbosity as V};

        let mut content = match ir {
            IR::Union {
                inputs: _,
                options: _,
            } => C::new("union"),
            IR::HConcat {
                inputs: _,
                schema: _,
                options: _,
            } => C::new("hconcat"),
            IR::Cache { input: _, id: _ } => C::new("cache"),
            IR::Filter {
                predicate,
                input: _,
            } => C::new("filter").arg(V::Default, d!(predicate)),
            #[cfg(feature = "python")]
            IR::PythonScan { options } => {
                use crate::plans::PythonPredicate;

                let predicate = match &options.predicate {
                    PythonPredicate::Polars(e) => d!(e),
                    PythonPredicate::PyArrow(s) => s.clone(),
                    PythonPredicate::None => "none".to_string(),
                };
                let with_columns = options
                    .with_columns
                    .as_ref()
                    .map_or(options.schema.len(), |v| v.len());
                let total_columns = options.schema.len();

                C::new("python scan")
                    .prop(V::Default, "π", format!("{with_columns}/{total_columns}"))
                    .prop(V::Default, "σ", predicate)
            },
            IR::Select {
                expr,
                input,
                schema,
                options: _,
            } => C::new("select").arg(V::Default, format!("{}/{}", expr.len(), schema.len())),
            IR::Sort {
                input: _,
                by_column,
                slice,
                sort_options: _,
            } => C::new("sort").prop(V::Default, "by", ds!(by_column)),
            IR::GroupBy {
                input,
                keys,
                aggs,
                maintain_order: _,
                schema: _,
                apply: _,
                options: _,
            } => {
                let keys = ds!(keys);
                let aggs = ds!(aggs);

                C::new("group by")
                    .prop(V::Default, "aggs", aggs)
                    .prop(V::Default, "by", keys)
            },
            IR::HStack { input, exprs, .. } => C::new("with columns").arg(V::Default, ds!(exprs)),
            IR::Slice { input, offset, len } => C::new("slice")
                .prop(V::Default, "offset", offset.to_string())
                .prop(V::Default, "len", len.to_string()),
            IR::Distinct { input, options } => {
                let mut content = C::new("distinct");

                if let Some(subset) = &options.subset {
                    content = content.prop(
                        V::Default,
                        "by",
                        TpgListBuilder::new().items(subset.iter().map(|v| v.to_string())),
                    );
                }

                content
            },
            IR::DataFrameScan {
                df,
                schema,
                output_schema,
            } => {
                let num_columns = output_schema.as_ref().map_or(schema.len(), |p| p.len());
                let total_columns = schema.len();

                C::new("table")
                    .prop(V::Default, "π", format!("{num_columns}/{total_columns}"))
                    .prop(V::Debug, "height", df.height())
            },
            IR::Scan {
                sources,
                file_info,
                hive_parts: _,
                predicate,
                scan_type,
                unified_scan_args,
                output_schema: _,
            } => {
                let total_columns =
                    file_info.schema.len() - usize::from(unified_scan_args.row_index.is_some());
                let with_columns = unified_scan_args
                    .projection
                    .as_ref()
                    .map_or(total_columns, |p| p.len());

                let title = match scan_type.as_ref() {
                    #[cfg(feature = "csv")]
                    FileScanIR::Csv { .. } => "csv scan",
                    #[cfg(feature = "json")]
                    FileScanIR::NDJson { .. } => "ndjson scan",
                    #[cfg(feature = "parquet")]
                    FileScanIR::Parquet { .. } => "parquet scan",
                    #[cfg(feature = "ipc")]
                    FileScanIR::Ipc { .. } => "ipc scan",
                    #[cfg(feature = "python")]
                    FileScanIR::PythonDataset { .. } => "python dataset scan",
                    FileScanIR::Anonymous { .. } => "anonymous scan",
                };

                let mut content = C::new(title)
                    .arg(
                        V::Default,
                        TpgListBuilder::new().items(sources.iter().map(|s| s.to_string())),
                    )
                    .prop(V::Default, "π", format!("{with_columns}/{total_columns}"));

                if let Some(predicate) = predicate.as_ref() {
                    content = content.prop(V::Default, "σ", d!(predicate).to_string());
                }

                if let Some(row_index) = unified_scan_args.row_index.as_ref() {
                    content = content.prop(
                        V::Default,
                        "row index",
                        format!("{} (+{})", row_index.name, row_index.offset),
                    );
                }
                content
            },
            IR::Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                use JoinType as T;
                let title = match options.args.how {
                    T::Inner => "inner join",
                    T::Left => "left join",
                    T::Right => "right join",
                    T::Full => "full join",
                    T::AsOf(_) => "asof join",
                    T::Semi => "semi join",
                    T::Anti => "anti join",
                    T::IEJoin => "in-equality join",
                    T::Cross => "cross join",
                };

                let mut content = C::new(title);

                if !left_on.is_empty() {
                    content = content.prop(V::Default, "left", ds!(left_on)).prop(
                        V::Default,
                        "right",
                        ds!(right_on),
                    );
                }

                content
            },
            IR::MapFunction { input, function } => C::new(function.to_string()),
            IR::ExtContext { input, .. } => C::new("external context"),
            IR::Sink { input, payload, .. } => {
                let title = match payload {
                    SinkTypeIR::Memory => "sink memory",
                    SinkTypeIR::File { .. } => "sink file",
                    SinkTypeIR::Partition { .. } => "sink partition",
                };
                C::new(title)
            },
            IR::SinkMultiple { .. } => unreachable!(),
            IR::SimpleProjection { input, columns } => {
                let num_columns = columns.as_ref().len();
                let total_columns = ir_arena.get(*input).schema(ir_arena).len();

                C::new("simple π")
                    .arg(V::Default, format!("{num_columns}/{total_columns}"))
                    .arg(
                        V::Default,
                        TpgListBuilder::new().items(columns.iter_names().map(|v| v.as_str())),
                    )
            },
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left,
                input_right,
                key,
            } => C::new("merge sorted").prop(V::Default, "on", key.as_str()),
            IR::Invalid => C::new("invalid"),
        };

        content.children = children;

        nodes.push(content);
    }

    TextPlanGraph {
        title: "IR".into(),
        roots,
        nodes,
        legend: Default::default(),
    }
}
