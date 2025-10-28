from pydantic import BaseModel, Field
from typing import List, Optional

# Each Pydantic model corresponds to a MongoDB collection
# Collection name is the lowercase class name


class Dataset(BaseModel):
    name: str = Field(..., description="Dataset name")
    header: List[str]
    sample_rows: List[List[str]]
    total_rows: Optional[int] = None


# Add other domain models as your app grows (e.g., Profile, Chart, Report)
