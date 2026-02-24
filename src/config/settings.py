from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class DataConfig:
    data_path: str
    date_column: str
    required_columns: List[str]
    sort_ascending: bool

    def __post_init__(self):
        if not self.data_path:
            raise ValueError("data_path cannot be empty.")
        if not self.date_column:
            raise ValueError("date_column cannot be empty.")
        if not self.required_columns:
            raise ValueError("required_columns cannot be empty.")
        if not isinstance(self.required_columns, list):
            raise ValueError("required_columns must be a list of column names.")
        
@dataclass(frozen=True)
class LabelConfig:
    gap_threshold: float
    use_percentage: bool

    def __post_init__(self):
        if self.gap_threshold <= 0:
            raise ValueError("gap_threshold must be positive.")
        
@dataclass(frozen=True)
class FeatureConfig:
    rolling_window: List[int]
    include_volume: bool
    feature_set_name: str

    def __post_init__(self):
        if not self.rolling_window:
            raise ValueError("rolling_window cannot be empty.")
        if not all(isinstance(w, int) and w > 0 for w in self.rolling_window):
            raise ValueError("all rolling window values must be positive integers.")
        if not self.feature_set_name:
            raise ValueError("feature_set_name cannot be empty.")
        
@dataclass(frozen=True)
class SplitConfig:
    split_ratio: Optional[float]
    split_date: Optional[str]
    train_label: str
    test_label: str

    def __post_init__(self):
        if self.split_ratio is None and self.split_date is None:
            raise ValueError("Either split_ratio or split_date must be provided.")

        if self.split_ratio is not None and self.split_date is not None:
            raise ValueError("Only one of split_ratio or split_date should be provided.")

        if self.split_ratio is not None:
            if not (0 < self.split_ratio < 1):
                raise ValueError("split_ratio must be between 0 and 1.")

        if not self.train_label:
            raise ValueError("train_label must be provided.")

        if not self.test_label:
            raise ValueError("test_label must be provided.")

        if self.train_label == self.test_label:
            raise ValueError("train_label and test_label must be different.")
        

@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    label: LabelConfig
    feature: FeatureConfig
    split: SplitConfig

    def __post_init__(self):
        # Cross-config validation

        # Example: ensure required columns contain essential OHLC fields
        required_price_fields = {"open", "close"}
        if not required_price_fields.issubset(set(self.data.required_columns)):
            raise ValueError(f"required_columns must include at least 'open' and 'close' fields for labeling.")
    