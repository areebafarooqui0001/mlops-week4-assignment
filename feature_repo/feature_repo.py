from feast import Entity, Feature, FeatureView, FileSource, ValueType
from datetime import timedelta

driver_stats = FileSource(
    path="data/driver_stats.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created"
)

driver = Entity(name="driver_id", value_type=ValueType.INT64, description="driver id")

driver_hourly_stats_view = FeatureView(
    name="driver_hourly_stats",
    entities=["driver_id"],
    ttl=timedelta(hours=1),
    features=[
        Feature(name="conv_rate", dtype=ValueType.FLOAT),
        Feature(name="acc_rate", dtype=ValueType.FLOAT),
        Feature(name="avg_daily_trips", dtype=ValueType.INT64),
    ],
    online=True,
    source=driver_stats,
    tags={"team": "driver_performance"},
)
