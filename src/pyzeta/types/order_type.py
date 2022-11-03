from __future__ import annotations

import typing
from dataclasses import dataclass

import borsh_construct as borsh
from anchorpy.borsh_extension import EnumForCodegen


class LimitJSON(typing.TypedDict):
    kind: typing.Literal["Limit"]


class PostOnlyJSON(typing.TypedDict):
    kind: typing.Literal["PostOnly"]


class FillOrKillJSON(typing.TypedDict):
    kind: typing.Literal["FillOrKill"]


@dataclass
class Limit:
    discriminator: typing.ClassVar = 0
    kind: typing.ClassVar = "Limit"

    @classmethod
    def to_json(cls) -> LimitJSON:
        return LimitJSON(
            kind="Limit",
        )

    @classmethod
    def to_encodable(cls) -> dict:
        return {
            "Limit": {},
        }


@dataclass
class PostOnly:
    discriminator: typing.ClassVar = 1
    kind: typing.ClassVar = "PostOnly"

    @classmethod
    def to_json(cls) -> PostOnlyJSON:
        return PostOnlyJSON(
            kind="PostOnly",
        )

    @classmethod
    def to_encodable(cls) -> dict:
        return {
            "PostOnly": {},
        }


@dataclass
class FillOrKill:
    discriminator: typing.ClassVar = 2
    kind: typing.ClassVar = "FillOrKill"

    @classmethod
    def to_json(cls) -> FillOrKillJSON:
        return FillOrKillJSON(
            kind="FillOrKill",
        )

    @classmethod
    def to_encodable(cls) -> dict:
        return {
            "FillOrKill": {},
        }


OrderTypeKind = typing.Union[Limit, PostOnly, FillOrKill]
OrderTypeJSON = typing.Union[LimitJSON, PostOnlyJSON, FillOrKillJSON]


def from_decoded(obj: dict) -> OrderTypeKind:
    if not isinstance(obj, dict):
        raise ValueError("Invalid enum object")
    if "Limit" in obj:
        return Limit()
    if "PostOnly" in obj:
        return PostOnly()
    if "FillOrKill" in obj:
        return FillOrKill()
    raise ValueError("Invalid enum object")


def from_json(obj: OrderTypeJSON) -> OrderTypeKind:
    if obj["kind"] == "Limit":
        return Limit()
    if obj["kind"] == "PostOnly":
        return PostOnly()
    if obj["kind"] == "FillOrKill":
        return FillOrKill()
    kind = obj["kind"]
    raise ValueError(f"Unrecognized enum kind: {kind}")


layout = EnumForCodegen(
    "Limit" / borsh.CStruct(),
    "PostOnly" / borsh.CStruct(),
    "FillOrKill" / borsh.CStruct(),
)
