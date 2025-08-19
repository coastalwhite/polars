pub mod dot;
pub mod json;
pub mod html;

#[derive(Clone, Copy)]
pub struct TpgKey(pub usize);

pub struct TextPlanGraph {
    pub title: Box<str>,

    pub roots: usize,
    pub nodes: Vec<TpgNode>,

    pub legend: Vec<Box<str>>,
}

pub struct TpgNode {
    pub title: Box<str>,

    pub tags: u64,

    arguments: Vec<(TpgVerbosity, TpgValue)>,
    properties: Vec<(TpgVerbosity, Box<str>, TpgValue)>,

    pub children: Vec<TpgKey>,
    pub subnodes: Option<TpgSubNodes>,
}

pub struct TpgSubNodes {
    pub start: TpgKey,
    pub roots: usize,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TpgVerbosity {
    Debug,
    #[default]
    Default,
    Necessary,
}

#[derive(Clone)]
pub enum TpgValue {
    Float(f64),
    Integer(i128),
    String(Box<str>),
    Binary(Box<[u8]>),
    List(Box<[TpgValue]>),
    Dictionary(Box<[(Box<str>, TpgValue)]>),
}

macro_rules! impl_value_from {
    ($($variant:ident: ($($t:ty),+)),+) => {
        $(
        $(
        impl From<$t> for TpgValue {
            fn from(v: $t) -> Self {
                Self::$variant(v.into())
            }
        }
        )+
        )+
    };
}

impl_value_from! {
    Float: (f32, f64),
    Integer: (i8, i16, i32, i64, i128, u8, u16, u32, u64),
    String: (&str, Box<str>, String),
    Binary: (&[u8], Box<[u8]>, Vec<u8>)
}

impl From<usize> for TpgValue {
    fn from(v: usize) -> Self {
        Self::Integer(v as i128)
    }
}

impl From<isize> for TpgValue {
    fn from(v: isize) -> Self {
        Self::Integer(v as i128)
    }
}

#[derive(Default, Clone)]
pub struct TpgListBuilder {
    values: Vec<TpgValue>,
}

#[derive(Default, Clone)]
pub struct TpgDictionaryBuilder {
    values: Vec<(Box<str>, TpgValue)>,
}

impl TpgListBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, v: impl Into<TpgValue>) {
        self.values.push(v.into());
    }

    pub fn extend<V: Into<TpgValue>>(&mut self, v: impl IntoIterator<Item = V>) {
        self.values.extend(v.into_iter().map(Into::into));
    }

    pub fn item(mut self, v: impl Into<TpgValue>) -> Self {
        self.push(v);
        self
    }

    pub fn items<V: Into<TpgValue>>(mut self, v: impl IntoIterator<Item = V>) -> Self {
        self.extend(v);
        self
    }

    pub fn finish(self) -> TpgValue {
        TpgValue::List(self.values.into_boxed_slice())
    }
}

impl TpgDictionaryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, key: impl Into<Box<str>>, value: impl Into<TpgValue>) {
        self.values.push((key.into(), value.into()));
    }

    pub fn extend<K: Into<Box<str>>, V: Into<TpgValue>>(
        &mut self,
        items: impl IntoIterator<Item = (K, V)>,
    ) {
        self.values
            .extend(items.into_iter().map(|(k, v)| (k.into(), v.into())));
    }

    pub fn item(mut self, key: impl Into<Box<str>>, value: impl Into<TpgValue>) -> Self {
        self.push(key, value);
        self
    }

    pub fn items<K: Into<Box<str>>, V: Into<TpgValue>>(
        mut self,
        items: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.extend(items);
        self
    }

    pub fn finish(self) -> TpgValue {
        TpgValue::Dictionary(self.values.into_boxed_slice())
    }
}

impl From<TpgListBuilder> for TpgValue {
    fn from(value: TpgListBuilder) -> Self {
        value.finish()
    }
}

impl From<TpgDictionaryBuilder> for TpgValue {
    fn from(value: TpgDictionaryBuilder) -> Self {
        value.finish()
    }
}

impl TpgNode {
    pub fn new(title: impl Into<Box<str>>) -> Self {
        Self {
            title: title.into(),
            tags: 0,
            arguments: Default::default(),
            properties: Default::default(),
            children: Default::default(),
            subnodes: None,
        }
    }

    pub fn push_argument(&mut self, verbosity: TpgVerbosity, value: impl Into<TpgValue>) {
        self.arguments.push((verbosity, value.into()));
    }

    pub fn arg(mut self, verbosity: TpgVerbosity, value: impl Into<TpgValue>) -> Self {
        self.push_argument(verbosity, value);
        self
    }

    pub fn push_property(
        &mut self,
        verbosity: TpgVerbosity,
        key: impl Into<Box<str>>,
        value: impl Into<TpgValue>,
    ) {
        self.properties.push((verbosity, key.into(), value.into()));
    }

    pub fn prop(
        mut self,
        verbosity: TpgVerbosity,
        key: impl Into<Box<str>>,
        value: impl Into<TpgValue>,
    ) -> Self {
        self.push_property(verbosity, key, value);
        self
    }
}
