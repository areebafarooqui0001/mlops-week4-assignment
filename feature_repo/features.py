from datetime import timedelta
from feast import Entity, FeatureView, Field, ValueType
from feast.types import Float32, String
from feast.infra.offline_stores.bigquery_source import BigQuerySource


iris_entity = Entity(name="iris_id", value_type=ValueType.INT64)

iris_bq_source = BigQuerySource(
    table="charged-muse-461217-u9.iris_feast.iris_data",
)

iris_fv = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_bq_source,
    tags={},
)

iris_target_fv = FeatureView(
    name="iris_target",
    entities=[iris_entity],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="species", dtype=String),
    ],
    source=iris_bq_source,
    tags={},
)