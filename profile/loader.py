"""
profile/loader.py — Load and validate the user profile from profile.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from core.exceptions import ProfileError
from core.logger import logger
from models.profile import UserProfile


def load_profile(profile_path: str = "./profile/profile.yaml") -> UserProfile:
    """
    Load the user profile from a YAML file and validate it.

    Raises ProfileError if the file is missing or structurally invalid.
    """
    path = Path(profile_path)

    if not path.exists():
        raise ProfileError(
            f"Profile file not found: {path.absolute()}\n"
            "Please copy profile/profile.yaml and fill in your details."
        )

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ProfileError(f"Profile file is empty: {path}")

    try:
        profile = UserProfile(**raw)
    except ValidationError as e:
        raise ProfileError(
            f"Profile validation failed:\n{e}"
        ) from e

    # Sanity warnings (not errors — agent can still run)
    if not profile.personal.email or "example.com" in profile.personal.email:
        logger.warning("⚠️  Profile: email is not set — applications may fail")

    if not profile.preferences.target_roles:
        logger.warning("⚠️  Profile: target_roles is empty — job scoring will be inaccurate")

    if not profile.skills.technical:
        logger.warning("⚠️  Profile: no technical skills listed — scoring will be imprecise")

    if not profile.experience:
        logger.warning("⚠️  Profile: no work experience listed")

    # Log resume location if specified in profile
    if profile.resume and profile.resume.default_resume:
        from pathlib import Path as _P
        rpath = _P(profile.resume.default_resume)
        if rpath.exists():
            logger.info("Resume found: {} ({:.0f} KB)", rpath.name, rpath.stat().st_size / 1024)
        else:
            logger.warning("⚠️  Resume path not found: {}", rpath)

    logger.info(
        "Profile loaded — name='{}' email='{}' roles={} skills={} experience={}",
        profile.personal.name,
        profile.personal.email,
        len(profile.preferences.target_roles),
        len(profile.skills.technical),
        len(profile.experience),
    )
    return profile
