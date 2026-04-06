from __future__ import annotations

import argparse
import sys

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.services.database import (
    Base,
    ChatMessage,
    ConversationCounter,
    Diary,
    KBSearchSession,
    MemorySnapshot,
    SessionState,
)
from src.services.database_config import (
    build_sync_database_url,
    build_sync_engine_kwargs,
    get_default_sqlite_path,
    is_sqlite_url,
)


MODEL_ORDER = [
    SessionState,
    MemorySnapshot,
    ConversationCounter,
    KBSearchSession,
    ChatMessage,
    Diary,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate AtriNexus SQLite data into PostgreSQL.")
    parser.add_argument(
        "--source",
        default=f"sqlite:///{get_default_sqlite_path()}",
        help="Source database URL. Defaults to local SQLite runtime DB.",
    )
    parser.add_argument(
        "--target",
        default=build_sync_database_url(),
        help="Target database URL. Defaults to ATRINEXUS_DATABASE_URL-derived sync URL.",
    )
    parser.add_argument(
        "--clear-target",
        action="store_true",
        help="Clear target tables before importing. Use with care.",
    )
    return parser.parse_args()


def ensure_target_is_postgres(target_url: str) -> None:
    if is_sqlite_url(target_url):
        raise SystemExit("Target database is still SQLite. Set ATRINEXUS_DATABASE_URL to a PostgreSQL URL first.")


def clone_instance(instance):
    values = {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }
    return instance.__class__(**values)


def count_rows(session, model) -> int:
    return len(session.execute(select(model)).scalars().all())


def clear_target(session) -> None:
    for model in reversed(MODEL_ORDER):
        session.query(model).delete()
    session.commit()


def migrate(source_url: str, target_url: str, clear_target_first: bool) -> None:
    ensure_target_is_postgres(target_url)

    source_engine = create_engine(source_url, **build_sync_engine_kwargs(source_url))
    target_engine = create_engine(target_url, **build_sync_engine_kwargs(target_url))

    Base.metadata.create_all(target_engine)

    SourceSession = sessionmaker(bind=source_engine, future=True)
    TargetSession = sessionmaker(bind=target_engine, future=True)

    with SourceSession() as source_session, TargetSession() as target_session:
        if clear_target_first:
            clear_target(target_session)
        else:
            existing = sum(count_rows(target_session, model) for model in MODEL_ORDER)
            if existing:
                raise SystemExit(
                    "Target database is not empty. Re-run with --clear-target if you really want to overwrite it."
                )

        for model in MODEL_ORDER:
            rows = source_session.execute(select(model)).scalars().all()
            for row in rows:
                target_session.merge(clone_instance(row))
            target_session.commit()
            print(f"{model.__tablename__}: migrated {len(rows)} rows")


def main() -> int:
    args = parse_args()
    migrate(args.source, args.target, args.clear_target)
    print("Migration completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
