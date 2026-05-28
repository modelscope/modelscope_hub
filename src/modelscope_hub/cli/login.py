"""``ms login`` and ``ms whoami`` commands.

The two flows share their HubApi construction but live in distinct
:class:`CLICommand` classes so each can be registered, tested and evolved
independently.
"""

from __future__ import annotations

import getpass
from argparse import _SubParsersAction

from .base import CLICommand, error, info, make_api, success


class LoginCommand(CLICommand):
    """Persist a token and verify it via ``GET /users/me``."""

    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "login",
            help="Authenticate with ModelScope Hub and persist the token locally.",
            description="Prompts for an API token (or accepts --token), then verifies it.",
        )
        parser.add_argument(
            "--token",
            dest="login_token",
            default=None,
            help="API token. If omitted, you will be prompted interactively.",
        )
        parser.set_defaults(_command=LoginCommand)

    def execute(self) -> None:
        token = self.args.login_token or getattr(self.args, "token", None)
        if not token:
            try:
                token = getpass.getpass("Token (input hidden): ")
            except (EOFError, KeyboardInterrupt):
                error("Login aborted.")
                raise SystemExit(130) from None
        if not token or not token.strip():
            error("A non-empty token is required.")
            raise SystemExit(2)

        api = make_api(self.args)
        user = api.login(token.strip())
        identity = user.username or user.email or str(user.id) or "<unknown>"
        success(f"Logged in as {identity}.")


class WhoamiCommand(CLICommand):
    """Show the currently authenticated user."""

    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "whoami",
            help="Show the user identified by the active token.",
        )
        parser.set_defaults(_command=WhoamiCommand)

    def execute(self) -> None:
        api = make_api(self.args)
        user = api.whoami()
        info(f"username   : {user.username or '-'}")
        info(f"email      : {user.email or '-'}")
        info(f"id         : {user.id if user.id is not None else '-'}")
        if user.description:
            info(f"description: {user.description}")
