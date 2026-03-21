class LabelEncoder:
    """Stores an ordered list of class names; picklable from any module."""

    def __init__(self, classes: list) -> None:
        self.classes_ = list(classes)

    def __repr__(self) -> str:
        return f"LabelEncoder(classes={self.classes_})"