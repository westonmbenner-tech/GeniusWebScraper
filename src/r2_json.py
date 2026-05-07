"""R2 JSON helpers for Lyric Atlas artifacts."""

from __future__ import annotations

import json
from typing import Any, TypeVar, cast

from botocore.exceptions import ClientError

from src.atlas_env import get_r2_bucket_name
from src.r2_client import build_r2_client

T = TypeVar("T")


def put_json_to_r2(key: str, data: Any) -> str:
    client = build_r2_client()
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    client.put_object(
        Bucket=get_r2_bucket_name(),
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )
    return key


def get_json_from_r2(key: str) -> T:
    client = build_r2_client()
    try:
        resp = client.get_object(Bucket=get_r2_bucket_name(), Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            raise FileNotFoundError(f"R2 object not found: {key}") from exc
        raise
    raw = resp["Body"].read()
    try:
        return cast(T, json.loads(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in R2 object: {key}") from exc


def delete_r2_object(key: str) -> None:
    client = build_r2_client()
    client.delete_object(Bucket=get_r2_bucket_name(), Key=key)


def list_r2_objects(prefix: str) -> list[str]:
    client = build_r2_client()
    keys: list[str] = []
    continuation: str | None = None
    while True:
        kwargs = {"Bucket": get_r2_bucket_name(), "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key:
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        continuation = resp.get("NextContinuationToken")
    return keys


def object_exists(key: str) -> bool:
    client = build_r2_client()
    try:
        client.head_object(Bucket=get_r2_bucket_name(), Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
