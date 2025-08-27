mod dot;
mod format;
pub mod inputs;
mod schema;
pub(crate) mod tree_format;

use std::borrow::Cow;
use std::fmt;

pub use dot::{EscapeLabel, IRDotDisplay, PathsDisplay, ScanSourcesDisplay};
pub use format::{ExprIRDisplay, IRDisplay, write_group_by, write_ir_non_recursive};
use polars_core::prelude::*;
use polars_utils::format_pl_smallstr;
use polars_utils::idx_vec::UnitVec;
use polars_utils::unique_id::UniqueId;
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use self::hive::HivePartitionsDf;
use crate::plans::set_order::InputOrder;
use crate::prelude::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IRPlan {
    pub lp_top: Node,
    pub lp_arena: Arena<IR>,
    pub expr_arena: Arena<AExpr>,
}

#[derive(Clone, Copy)]
pub struct IRPlanRef<'a> {
    pub lp_top: Node,
    pub lp_arena: &'a Arena<IR>,
    pub expr_arena: &'a Arena<AExpr>,
}

/// [`IR`] is a representation of [`DslPlan`] with [`Node`]s which are allocated in an [`Arena`]
/// In this IR the logical plan has access to the full dataset.
#[derive(Clone, Debug, Default, IntoStaticStr)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum IR {
    #[cfg(feature = "python")]
    PythonScan {
        options: PythonOptions,
    },
    Slice {
        input: Node,
        offset: i64,
        len: IdxSize,
    },
    Filter {
        input: Node,
        predicate: ExprIR,
    },
    Scan {
        sources: ScanSources,
        file_info: FileInfo,
        hive_parts: Option<HivePartitionsDf>,
        predicate: Option<ExprIR>,
        /// schema of the projected file
        output_schema: Option<SchemaRef>,
        scan_type: Box<FileScanIR>,
        /// generic options that can be used for all file types.
        unified_scan_args: Box<UnifiedScanArgs>,
    },
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        // Schema of the projected file
        // If `None`, no projection is applied
        output_schema: Option<SchemaRef>,
    },
    // Only selects columns (semantically only has row access).
    // This is a more restricted operation than `Select`.
    SimpleProjection {
        input: Node,
        columns: SchemaRef,
    },
    // Polars' `select` operation. This may access full materialized data.
    Select {
        input: Node,
        expr: Vec<ExprIR>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Sort {
        input: Node,
        by_column: Vec<ExprIR>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },
    Cache {
        input: Node,
        /// This holds the `Arc<DslPlan>` to guarantee uniqueness.
        id: UniqueId,
    },
    GroupBy {
        input: Node,
        keys: Vec<ExprIR>,
        aggs: Vec<ExprIR>,
        schema: SchemaRef,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
        apply: Option<PlanCallback<DataFrame, DataFrame>>,
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: SchemaRef,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        options: Arc<JoinOptionsIR>,
    },
    HStack {
        input: Node,
        exprs: Vec<ExprIR>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Distinct {
        input: Node,
        options: DistinctOptionsIR,
    },
    MapFunction {
        input: Node,
        function: FunctionIR,
    },
    Union {
        inputs: Vec<Node>,
        options: UnionOptions,
    },
    /// Horizontal concatenation
    /// - Invariant: the names will be unique
    HConcat {
        inputs: Vec<Node>,
        schema: SchemaRef,
        options: HConcatOptions,
    },
    ExtContext {
        input: Node,
        contexts: Vec<Node>,
        schema: SchemaRef,
    },
    Sink {
        input: Node,
        payload: SinkTypeIR,
    },
    /// Node that allows for multiple plans to be executed in parallel with common subplan
    /// elimination and everything.
    SinkMultiple {
        inputs: Vec<Node>,
    },
    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        input_left: Node,
        input_right: Node,
        key: PlSmallStr,
    },
    #[default]
    Invalid,
}

impl IRPlan {
    pub fn new(top: Node, ir_arena: Arena<IR>, expr_arena: Arena<AExpr>) -> Self {
        Self {
            lp_top: top,
            lp_arena: ir_arena,
            expr_arena,
        }
    }

    pub fn root(&self) -> &IR {
        self.lp_arena.get(self.lp_top)
    }

    pub fn as_ref(&self) -> IRPlanRef<'_> {
        IRPlanRef {
            lp_top: self.lp_top,
            lp_arena: &self.lp_arena,
            expr_arena: &self.expr_arena,
        }
    }

    pub fn describe(&self) -> String {
        self.as_ref().describe()
    }

    pub fn describe_tree_format(&self) -> String {
        self.as_ref().describe_tree_format()
    }

    pub fn display(&self) -> format::IRDisplay<'_> {
        self.as_ref().display()
    }

    pub fn display_dot(&self) -> dot::IRDotDisplay<'_> {
        self.as_ref().display_dot()
    }
}

impl<'a> IRPlanRef<'a> {
    pub fn root(self) -> &'a IR {
        self.lp_arena.get(self.lp_top)
    }

    pub fn with_root(self, root: Node) -> Self {
        Self {
            lp_top: root,
            lp_arena: self.lp_arena,
            expr_arena: self.expr_arena,
        }
    }

    pub fn display(self) -> format::IRDisplay<'a> {
        format::IRDisplay::new(self)
    }

    pub fn display_dot(self) -> dot::IRDotDisplay<'a> {
        dot::IRDotDisplay::new(self)
    }

    pub fn describe(self) -> String {
        self.display().to_string()
    }

    pub fn describe_tree_format(self) -> String {
        let mut visitor = tree_format::TreeFmtVisitor::default();
        tree_format::TreeFmtNode::root_logical_plan(self).traverse(&mut visitor);
        format!("{visitor:#?}")
    }
}

pub fn remove_ordering_requirements(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    orders: &mut PlHashMap<Node, set_order::PortOrder>,
) {
    let mut outputs = PlHashMap::default();

    // Get the per-node outputs and leaves
    {
        let mut stack = Vec::new();

        for root in roots {
            assert!(matches!(ir_arena.get(*root), IR::Sink { .. }));
            outputs.insert(*root, Vec::new());
            stack.extend(
                ir_arena
                    .get(*root)
                    .inputs()
                    .enumerate()
                    .map(|(i, node)| (*root, i, node)),
            );
        }

        while let Some((input, i, node)) = stack.pop() {
            let outputs = outputs.entry(node).or_default();
            let has_been_visisited_before = !outputs.is_empty();
            outputs.push((i, input));

            let ir = ir_arena.get(node);
            if has_been_visisited_before {
                continue;
            }

            stack.extend(ir.inputs().enumerate().map(|(i, input)| (node, i, input)));
        }
    }

    let mut visited = PlHashMap::new();
    for root in roots {
        remove_ordering_requirements_rec(
            *root,
            ir_arena,
            expr_arena,
            orders,
            &mut visited,
            &outputs,
        );
    }
}

#[recursive::recursive]
pub fn remove_ordering_requirements_rec(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    orders: &mut PlHashMap<Node, set_order::PortOrder>,
    visited: &mut PlHashMap<Node, Arc<Schema>>,
    outputs: &PlHashMap<Node, Vec<(usize, Node)>>,
) {
    if visited.contains_key(&node) {
        return;
    }

    static ROW_ORDER_COLUMN_NAME: &str = "__PL_ROWORDER";
    const RO_PLSTR: PlSmallStr = PlSmallStr::from_static(ROW_ORDER_COLUMN_NAME);

    fn get_dtype(row_order_width: usize) -> DataType {
        match row_order_width {
            0 => unreachable!("row_index_width == 0"),
            1 => IDX_DTYPE,
            width => DataType::Array(Box::new(IDX_DTYPE), width),
        }
    }
    fn get_width(schema: &Schema) -> usize {
        assert!(!schema.is_empty());
        let (name, dtype) = schema.get_at_index(schema.len() - 1).unwrap();
        assert_eq!(name, &RO_PLSTR);
        match dtype {
            dtype if dtype == &IDX_DTYPE => 1,
            DataType::Array(inner, width) if inner.as_ref() == &IDX_DTYPE => *width,
            _ => unreachable!("invalid row order dtype {dtype}"),
        }
    }

    fn insert_into_schema(schema: &Schema, row_order_width: usize) -> Schema {
        let dtype = get_dtype(row_order_width);
        let mut schema = schema.clone();
        schema.try_insert(RO_PLSTR, dtype.clone()).unwrap();
        schema
    }

    let original_inputs = ir_arena.get(node).inputs().collect::<Vec<_>>();
    for input in original_inputs.iter().copied() {
        remove_ordering_requirements_rec(input, ir_arena, expr_arena, orders, visited, outputs);
    }

    let mut original_order = None;
    if let IR::Join { options, .. } = ir_arena.get_mut(node) {
        let order = orders.get_mut(&node).unwrap();
        if order.propagates_order() {
            original_order = Some(options.args.maintain_order);
            if !matches!(options.args.maintain_order, MaintainOrderJoin::None) {
                let mut new_options = options.as_ref().clone();
                new_options.args.maintain_order = MaintainOrderJoin::None;
                *options = Arc::new(new_options);
            }

            order.inputs[0] = InputOrder::Preserving;
            order.inputs[1] = InputOrder::Preserving;
        }
    }

    let order = &orders[&node];
    let schema = if !order.propagates_order() {
        ir_arena.get(node).schema(ir_arena).into_owned()
    } else {
        for (i, (input, order)) in original_inputs
            .into_iter()
            .zip(order.inputs.iter())
            .enumerate()
        {
            if order.needs_ordered() {
                let sorted_input = ir_arena.add(IR::Sort {
                    input,
                    by_column: vec![AExprBuilder::col(RO_PLSTR, expr_arena).expr_ir(RO_PLSTR)],
                    slice: None,
                    sort_options: Default::default(),
                });
                *ir_arena.get_mut(node).inputs_mut().nth(i).unwrap() = sorted_input;
            }
        }

        if !order.inputs.iter().any(|i| i.may_propagate())
            && order.output_ordered.iter().any(|o| *o)
        {
            let output_schema = ir_arena.get(node).schema(ir_arena);
            let output_schema = Arc::new(insert_into_schema(output_schema.as_ref().as_ref(), 1));
            for (&is_ordered, (i, output)) in order.output_ordered.iter().zip(outputs[&node].iter())
            {
                if is_ordered {
                    let input = ir_arena.get(*output).inputs().nth(*i).unwrap();
                    let row_index_input = ir_arena.add(IR::HStack {
                        input,
                        exprs: vec![AExprBuilder::row_index(expr_arena).expr_ir(RO_PLSTR)],
                        schema: output_schema.clone(),
                        options: Default::default(),
                    });
                    *ir_arena.get_mut(*output).inputs_mut().nth(*i).unwrap() = row_index_input;
                }
            }
        }

        match ir_arena.get(node) {
            IR::Slice { input, .. } | IR::Filter { input, .. } | IR::Cache { input, .. } => {
                // Nothing needs to happen for these nodes. They will automatically propagate their
                // input.
                ir_arena.get(*input).schema(ir_arena).into_owned()
            },

            IR::SimpleProjection { input, columns } => {
                let input_schema = ir_arena.get(*input).schema(ir_arena);
                let width = get_width(&input_schema);

                let IR::SimpleProjection { columns, .. } = ir_arena.get_mut(node) else {
                    unreachable!();
                };
                let schema = Arc::new(insert_into_schema(columns, width));
                *columns = schema.clone();
                schema
            },
            IR::HStack {
                input,
                exprs,
                schema,
                ..
            } => {
                // lf.with_columns(key = expr, ...)
                //   to
                // lf.with_columns(key = expr, ...)
                //   .select([..., ROW_ORDER])

                let input_schema = ir_arena.get(*input).schema(ir_arena);
                let width = get_width(&input_schema);
                let num_new_columns = exprs
                    .iter()
                    .filter(|e| input_schema.contains(e.output_name().as_str()))
                    .count();

                let IR::HStack { schema, .. } = ir_arena.get_mut(node) else {
                    unreachable!();
                };

                let output_schema = Arc::new(insert_into_schema(schema.as_ref(), width));
                if num_new_columns == 0 {
                    *schema = output_schema.clone();
                } else {
                    *schema = Arc::new(
                        schema
                            .new_inserting_at_index(
                                schema.len() - num_new_columns,
                                RO_PLSTR,
                                get_dtype(width),
                            )
                            .unwrap(),
                    );

                    let f = ir_arena.take(node);
                    let new_node = ir_arena.add(f);
                    ir_arena.replace(
                        node,
                        IR::SimpleProjection {
                            input: new_node,
                            columns: output_schema.clone(),
                        },
                    );
                }
                output_schema
            },
            IR::Select {
                input,
                expr,
                schema,
                ..
            } => {
                let input_schema = ir_arena.get(*input).schema(ir_arena);
                let width = get_width(&input_schema);
                if cfg!(debug_assertions) {
                    let mut has_elementwise = false;
                    for e in expr.iter() {
                        has_elementwise |= is_elementwise_rec(e.node(), expr_arena);
                    }
                    assert!(has_elementwise);
                }

                let IR::Select { schema, expr, .. } = ir_arena.get_mut(node) else {
                    unreachable!();
                };
                let output_schema = Arc::new(insert_into_schema(schema.as_ref(), width));
                *schema = output_schema.clone();
                expr.push(AExprBuilder::col(RO_PLSTR, expr_arena).expr_ir(RO_PLSTR));
                output_schema
            },
            IR::GroupBy {
                input,
                keys,
                aggs,
                schema,
                maintain_order,
                options,
                apply,
            } => {
                // lf.group_by(keys).agg(aggs)
                //   to
                // lf.group_by(keys).agg(aggs + [ROW_ORDER.first()])

                assert!(apply.is_none());

                let input_schema = ir_arena.get(*input).schema(ir_arena);
                let width = get_width(&input_schema);
                let output_schema = Arc::new(insert_into_schema(schema.as_ref(), width));

                let IR::GroupBy { schema, aggs, .. } = ir_arena.get_mut(node) else {
                    unreachable!();
                };
                *schema = output_schema.clone();
                aggs.push(
                    AExprBuilder::col(RO_PLSTR, expr_arena)
                        .first(expr_arena)
                        .expr_ir(RO_PLSTR),
                );
                output_schema
            },
            IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } => {
                use MaintainOrderJoin as MOJ;
                let mut schema_left = visited[input_left].clone();
                let mut schema_right = visited[input_right].clone();

                let (input_left, input_right) = (*input_left, *input_right);

                let num_preserving = order.inputs.iter().filter(|i| !i.is_unordered()).count();
                assert!(num_preserving > 0);

                let mut row_orders = Vec::new();
                if num_preserving > 1 {
                    use MaintainOrderJoin as MOJ;
                    let inputs = match original_order {
                        Some(MOJ::LeftRight) => [input_left, input_right],
                        Some(MOJ::RightLeft) => [input_right, input_left],
                        _ => unreachable!(),
                    };

                    for (i, (input, schema)) in inputs
                        .into_iter()
                        .zip([&mut schema_left, &mut schema_right])
                        .enumerate()
                    {
                        let name = format_pl_smallstr!("{ROW_ORDER_COLUMN_NAME}_{i}");
                        let renamed = AExprBuilder::col(RO_PLSTR, expr_arena).expr_ir(name.clone());
                        row_orders.push(name);
                        let new_input = IRBuilder::new(input, expr_arena, ir_arena)
                            .with_columns(vec![renamed], Default::default())
                            .drop([RO_PLSTR])
                            .node();

                        *ir_arena.get_mut(node).inputs_mut().nth(i).unwrap() = new_input;
                    }
                }

                let IR::Join {
                    schema,
                    left_on,
                    right_on,
                    options,
                    ..
                } = ir_arena.get_mut(node)
                else {
                    unreachable!();
                };
                *schema = det_join_schema(
                    &schema_left,
                    &schema_right,
                    left_on,
                    right_on,
                    options,
                    expr_arena,
                )
                .unwrap();

                if !row_orders.is_empty() {
                    let f = ir_arena.take(node);
                    let new_node = ir_arena.add(f);

                    let concat = AExprBuilder::concat_arr(
                        row_orders
                            .iter()
                            .map(|n| AExprBuilder::col(n.clone(), expr_arena).expr_ir(n.clone()))
                            .collect(),
                        expr_arena,
                    )
                    .expr_ir(RO_PLSTR);
                    let with_row_order = IRBuilder::new(new_node, expr_arena, ir_arena)
                        .with_columns(vec![concat], Default::default())
                        .drop(row_orders)
                        .build();
                    ir_arena.replace(node, with_row_order);
                }

                ir_arena.get(node).schema(ir_arena).into_owned()
            },

            IR::MapFunction { input, function } => {
                // lf.f()
                //   to
                // lf.f()
                //   .with_columns(ROW_ORDER = pl.concat_arr([ROW_ORDER, row_index()]))
                //   .select([..., ROW_ORDER])

                let input_schema = ir_arena.get(*input).schema(ir_arena);
                let mut output_schema = function.schema(&input_schema).unwrap().into_owned();

                let input = *input;

                let f = ir_arena.take(node);
                let new_node = ir_arena.add(f);

                let row_order = AExprBuilder::col(RO_PLSTR, expr_arena);
                let row_index = AExprBuilder::row_index(expr_arena);
                let new_row_order = AExprBuilder::concat_arr(
                    vec![
                        row_order.expr_ir_infer_name(expr_arena),
                        row_index
                            .cast(DataType::Array(Box::new(IDX_DTYPE), 1), expr_arena)
                            .expr_ir_infer_name(expr_arena),
                    ],
                    expr_arena,
                );

                let with_columns = IR::HStack {
                    input: new_node,
                    schema: output_schema.clone(),
                    exprs: vec![new_row_order.expr_ir(RO_PLSTR)],
                    options: Default::default(),
                };

                let mut propagated_node = with_columns;
                if output_schema
                    .get_at_index(output_schema.len() - 1)
                    .unwrap()
                    .0
                    != &RO_PLSTR
                {
                    let mut fixed_schema = output_schema.as_ref().clone();
                    let dtype = fixed_schema.shift_remove(ROW_ORDER_COLUMN_NAME).unwrap();
                    fixed_schema.try_insert(RO_PLSTR, dtype).unwrap();
                    output_schema = Arc::new(fixed_schema);

                    propagated_node = IR::SimpleProjection {
                        input: ir_arena.add(propagated_node),
                        columns: output_schema.clone(),
                    };
                }
                ir_arena.replace(node, propagated_node);
                output_schema
            },
            IR::Union {
                inputs, options, ..
            } => {
                // concat(lfs)
                //   to
                // concat([
                //   lf[i].with_column(ROW_ORDER = pl.concat_arr([[i], ROW_ORDER]))
                //   if does_input_preserve_order[i]
                //   else
                //   lf[i].with_column(ROW_ORDER = [i] + [0] * (row_order_width - 1))
                //   for i, lf in enumerate(lfs)
                // ])

                let mut max_width = 0;
                for (order, input) in order.inputs.iter().zip(inputs.iter()) {
                    if order.may_propagate() {
                        max_width = max_width.max(get_width(&visited[input]));
                    }
                }
                assert!(max_width > 0);

                let mut inputs = inputs.clone();
                for (i, (input, order)) in inputs.iter_mut().zip(order.inputs.iter()).enumerate() {
                    let input_schema = &visited[input];
                    let input_idx = i as IdxSize;
                    use InputOrder as I;
                    let input_row_order_width = match order {
                        I::Unordered => 0,
                        I::Preserving | I::Observing => get_width(&input_schema),
                        _ => unreachable!(),
                    };

                    let mut row_order = vec![0 as IdxSize; max_width - input_row_order_width];
                    row_order[0] = input_idx;
                    let row_order = Series::new(PlSmallStr::EMPTY, row_order);
                    let row_order = Scalar::new_array(row_order, 1);
                    let mut expr = AExprBuilder::lit_scalar(row_order, expr_arena);

                    if input_row_order_width > 0 {
                        let existing_row_order =
                            AExprBuilder::col(RO_PLSTR, expr_arena).expr_ir(RO_PLSTR);
                        expr = AExprBuilder::concat_arr(
                            vec![expr.expr_ir_infer_name(expr_arena), existing_row_order],
                            expr_arena,
                        );
                    }
                    let expr = expr.expr_ir(RO_PLSTR);

                    *input = ir_arena.add(IR::HStack {
                        input: *input,
                        exprs: vec![expr],
                        schema: input_schema.clone(),
                        options: Default::default(),
                    });
                }

                let IR::Union { inputs: i, .. } = ir_arena.get_mut(node) else {
                    unreachable!();
                };
                *i = inputs;

                ir_arena.get(node).schema(ir_arena).into_owned()
            },

            IR::Distinct { input, options } => {
                let schema = ir_arena.get(*input).schema(ir_arena).into_owned();
                let IR::Distinct { options, .. } = ir_arena.get_mut(node) else {
                    unreachable!();
                };
                options.keep_strategy = UniqueKeepStrategy::First;
                schema
            },

            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted { .. } => unreachable!("merge_sorted never propagates order"),
            #[cfg(feature = "python")]
            IR::PythonScan { .. } => unreachable!("python_scan never propagates order"),

            IR::HConcat { .. }
            | IR::Scan { .. }
            | IR::DataFrameScan { .. }
            | IR::Sink { .. }
            | IR::SinkMultiple { .. }
            | IR::Sort { .. } => {
                unreachable!("these nodes never have to propagate ordering")
            },

            IR::ExtContext { .. } | IR::Invalid => unreachable!(),
        }
    };

    visited.insert(node, schema.clone());
}

impl fmt::Debug for IRPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <format::IRDisplay as fmt::Display>::fmt(&self.display(), f)
    }
}

impl fmt::Debug for IRPlanRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <format::IRDisplay as fmt::Display>::fmt(&self.display(), f)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // skipped for now
    #[ignore]
    #[test]
    fn test_alp_size() {
        assert!(size_of::<IR>() <= 152);
    }
}
