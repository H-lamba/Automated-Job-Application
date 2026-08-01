import json
from pathlib import Path
from datetime import datetime, timezone
from models.job import JobListing
from llm.response_parser import ExtractedForm

class FabricationStore:
    def __init__(self, base_dir: str = "./data/fabricated"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_previous_answers(self) -> dict[str, str]:
        """Loads all past fabricated answers to ensure consistency."""
        answers = {}
        for file in self.base_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                for field in data.get("fields", []):
                    if field.get("label") and field.get("answer"):
                        answers[field["label"]] = field["answer"]
            except Exception:
                pass
        return answers

    def save(self, job: JobListing, form: ExtractedForm) -> Path:
        fabricated_only = [f for f in form.fields if f.source == "fabricated"]
        if not fabricated_only:
            return None
            
        payload = {
            "job_id": job.id,
            "company": job.company,
            "title": job.title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fields": [f.model_dump() for f in fabricated_only],
        }
        path = self.base_dir / f"{job.id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
