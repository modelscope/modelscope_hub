"""Tests for tolerant user profile parsing."""

from __future__ import annotations

import pytest

from modelscope_hub.api import HubApi
from modelscope_hub.types import UserInfo


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "Username": "legacy-user",
                "UserId": 123,
                "Email": "legacy@example.com",
                "Avatar": "https://avatar.example/legacy.png",
                "Description": "legacy description",
            },
            UserInfo(
                id=123,
                username="legacy-user",
                email="legacy@example.com",
                avatar_url="https://avatar.example/legacy.png",
                description="legacy description",
            ),
        ),
        (
            {
                "Name": "pre-user",
                "sub": "sub-123",
                "email": "pre@example.com",
                "picture": "https://avatar.example/pre.png",
                "description": "pre description",
            },
            UserInfo(
                id="sub-123",
                username="pre-user",
                email="pre@example.com",
                avatar_url="https://avatar.example/pre.png",
                description="pre description",
            ),
        ),
        (
            {
                "preferred_username": "display-name",
                "name": "login-handle",
                "user_id": "uid-1",
                "mail": "mail@example.com",
                "avatarUrl": "https://avatar.example/a.png",
                "bio": "bio text",
            },
            UserInfo(
                id="uid-1",
                username="login-handle",
                email="mail@example.com",
                avatar_url="https://avatar.example/a.png",
                description="bio text",
            ),
        ),
        ({"preferred_username": "fallback-user", "ID": 0}, UserInfo(id=0, username="fallback-user")),
    ],
)
def test_user_info_accepts_legacy_and_oidc_field_names(payload, expected):
    assert UserInfo.from_dict(payload) == expected


def test_user_info_ignores_empty_aliases_until_a_non_empty_value():
    user = UserInfo.from_dict(
        {
            "Username": "",
            "username": None,
            "name": "resolved-user",
            "Email": "",
            "email": "resolved@example.com",
        }
    )

    assert user.username == "resolved-user"
    assert user.email == "resolved@example.com"


def test_whoami_uses_user_info_field_compatibility(monkeypatch):
    api = HubApi(token="token-for-test")
    monkeypatch.setattr(
        api.openapi,
        "get_current_user",
        lambda: {
            "Name": "pre-user",
            "sub": "sub-123",
            "email": "pre@example.com",
            "description": "pre description",
        },
    )

    user = api.whoami()

    assert user.username == "pre-user"
    assert user.id == "sub-123"
    assert user.email == "pre@example.com"
    assert user.description == "pre description"
