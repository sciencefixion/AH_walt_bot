from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

class PassageModel(BaseModel):
    id: Annotated[int, Field(gt=0)]
    journal_id: Annotated[int, Field(gt=0)]
    title: Annotated[str, Field(min_length=1)] = None
    content: Annotated[str, Field(min_length=1, max_length=500)]
    created_at: Annotated[datetime, Field(gt=0)]
    updated_at: Annotated[datetime, Field(gt=0)]= None