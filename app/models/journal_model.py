from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

class JournalModel(BaseModel):
    id: Annotated[int, Field(gt=0)]
    title: Annotated[str, Field(min_length=1, max_length=160)]
    created_at: Annotated[datetime, Field(gt=0)]
    updated_at: Annotated[datetime, Field(gt=0)] = None