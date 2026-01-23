from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.models.journal_model import JournalModel

router = APIRouter(
    prefix="/journals",
    tags=["journals"]
)

# temporary db
journal_database = {
    1: JournalModel(
        id=1,
        title="Dream Journal",
        created_at=datetime(2024, 1, 2, 10, 30),  # January 2, 2026, at 10:30 AM
    ),
    2: JournalModel(
        id=2,
        title="Daily TODO List",
        created_at=datetime(2025, 1, 7, 11, 15)
    ),
    3: JournalModel(
        id=3,
        title="Shower Thoughts",
        created_at=datetime(2025, 11, 12, 8, 30)
    )
}


# Create/PUT journal -
@router.post("/", status_code=201)
async def create_journal(journal: JournalModel):
    for existing_journal in journal_database.values():
        if existing_journal.title == journal.title:
            raise HTTPException(
                status_code=400, detail="Journal title already taken! Choose another."
            )

    journal.id = len(journal_database) + 1

    journal_database[journal.id] = journal

    return {
        "message": journal.title + " created successfully!",
        "inserted_journal": journal,
    }


# Get all journals
@router.get("/")
async def get_all_journals():
    # TODO: maybe later if db is empty raise an exception
    return journal_database

# Get journal by id
@router.get("/journals/{journal_id}")
async def get_journal(journal_id: int):
    if journal_id in journal_database:
        return journal_database[journal_id]
    else:
        raise HTTPException(
            status_code=404, detail="Journal ID not found - cannot retrieve!"
        )


# DELETE journal by id
@router.delete("/journals/{journal_id}")
async def delete_journal(journal_id: int):
    if journal_id in journal_database:
        deleted_journal = journal_database.pop(journal_id)
        return {
            "message": f"Journal {deleted_journal.title} deleted successfully!",
            "deleted_journal": deleted_journal,
        }
    else:
        raise HTTPException(
            status_code=404, detail="Journal ID not found - cannot delete!"
        )


# PATCH update journal by id -
@router.put("/{journal_id}")
async def update_journal_info(journal_id: int, updated_journal: JournalModel):
    if journal_id in journal_database:
        journal_database[journal_id].title = updated_journal.title
        journal_database[journal_id].updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "message": f"{journal_database[journal_id].title} updated successfully!",
            "updated_journal": journal_database[journal_id],
        }
    else:
        raise HTTPException(
            status_code=404, detail="Journal ID not found - cannot update!"
        )

